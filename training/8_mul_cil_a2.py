# -*- coding: utf-8 -*-

# Copyright (c) 2025 Endang Rusiana.
# This work is licensed under the terms of the MIT License.
# For a copy, see <https://opensource.org/licenses/MIT>.
import os
import glob
import time
import gc
import psutil
import numpy as np
import tensorflow as tf
import argparse
from tensorflow.keras.callbacks import CSVLogger, Callback
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import mixed_precision
import random
import h5py
import csv
from tensorflow.keras import backend as K
from tensorflow.keras.initializers import Constant

mixed_precision.set_global_policy('mixed_float16')
# mixed_precision.set_global_policy('float32') 

# === Konfigurasi ===
# With Shuffle
# With LayerNorm
dataset_dir = "/opt/carla-simulator/PythonAPI/examples/dataset_saved"
batch_size = 120
h5_files = sorted(glob.glob(os.path.join(dataset_dir, "cil_batch_*.h5")))

# === Seed ===
np.random.seed(42)
tf.random.set_seed(42)

def load_and_index_h5_files(h5_file_paths):
    """
    Membuka file HDF5 dan membuat indeks per-command (LANEFOLLOW, LEFT, RIGHT, STRAIGHT)
    """
    opened_files = []
    index_per_cmd = {"LANEFOLLOW": [], "LEFT": [], "RIGHT": [], "STRAIGHT": []}

    for file_idx, path in enumerate(h5_file_paths):
        f = h5py.File(path, 'r')
        opened_files.append(f)

        road_options = f['road_option'][:]
        for i, opt in enumerate(road_options):
            cmd = opt.decode('utf-8') if isinstance(opt, bytes) else str(opt)
            if cmd in index_per_cmd:
                index_per_cmd[cmd].append((file_idx, i))

    # 🔹 Shuffle
    for cmd in ("LANEFOLLOW","LEFT", "RIGHT", "STRAIGHT"):
        random.shuffle(index_per_cmd[cmd])

    return opened_files, index_per_cmd

def simple_balanced_tf_generator(batch_size, index_per_cmd, opened_files,
                                 shuffle_triplets=True, shuffle_full=False, seed=12345):
    import numpy as np
    rng = np.random.default_rng(seed)

    steer_aug = 0.15

    lf_data = sorted(index_per_cmd["LANEFOLLOW"], key=lambda x: (x[0], x[1]))
    lf_ptr, max_lf = 0, len(lf_data)
    other_cmds = ["LEFT", "RIGHT", "STRAIGHT"]
    pools = {cmd: index_per_cmd[cmd] for cmd in other_cmds}
    ptrs  = {cmd: 0 for cmd in other_cmds}
    dup_ptr_global = 0
    cmd_to_idx = {"LANEFOLLOW": 0, "LEFT": 1, "RIGHT": 2, "STRAIGHT": 3}
    views = [("left", +steer_aug, 0), ("center", 0.0, 1), ("right", -steer_aug, 2)]
    # views = [("center", 0.0, 1)]
    num_views = len(views)
    print(f"num views:{num_views}")
    center_idx_in_block = 0

    # base_per_cmd = max(1, batch_size // (4 * 3))
    base_per_cmd = max(1, batch_size // (4 * num_views))

    while lf_ptr + base_per_cmd <= max_lf and all(len(pools[c]) > 0 for c in other_cmds):
        batch_entries = []

        # basis LF
        for _ in range(base_per_cmd):
            batch_entries.append(("LANEFOLLOW", lf_data[lf_ptr]))
            lf_ptr += 1

        # basis minor (cyclic)
        for cmd in other_cmds:
            for _ in range(base_per_cmd):
                j = ptrs[cmd]
                idx = pools[cmd][j % len(pools[cmd])]
                ptrs[cmd] = j + 1
                batch_entries.append((cmd, idx))

        # (opsional) acak urutan basis (triplet) SEBELUM expand 3-view
        if shuffle_triplets and not shuffle_full:
            rng.shuffle(batch_entries)

        X_rgbd, X_speed, X_cmd, Y_out, Y_speed = [], [], [], [], []
        for cmd, (file_idx, i) in batch_entries:
            f = opened_files[file_idx]
            steer = float(f['steer'][i])
            throttle = float(f['throttle'][i])
            brake = float(f['brake'][i])
            speed = ((float(f['speed'][i]) / 30.0) * 2.0) - 1.0 #normalisasi ke [-1,1]

            rgb_all = f['rgb'][i]     # (3,H,W,3)
            depth_all = f['depth'][i] # (3,H,W)

            for _, offset, idx_view in views:
                steer_new = np.clip(steer + offset, -1.0, 1.0)
                rgb = rgb_all[idx_view]
                depth = depth_all[idx_view]
                # rgbd = np.concatenate([rgb, depth[..., None]], axis=-1).astype(np.float32) / 255.0
                rgbd = np.concatenate([rgb, depth[..., None]], axis=-1).astype(np.float32)
                rgbd = (rgbd - 127.5) / 127.5   # → [-1, 1]

                onehot = np.zeros(4, dtype=np.float32)
                onehot[cmd_to_idx[cmd]] = 1.0

                X_rgbd.append(rgbd)
                X_speed.append([speed])
                X_cmd.append(onehot)
                Y_out.append([steer_new, throttle, brake])
                Y_speed.append([speed])

        total = len(X_rgbd)
        if total > batch_size:
            target_per_cmd = batch_size // (4 * 3)
            keep_indices, base_offset = [], 0
            kept = {k: 0 for k in cmd_to_idx.keys()}

            # pilih tepat 'target_per_cmd' basis per command (masih menjaga balancing)
            for bi, (cmd, _) in enumerate(batch_entries):
                trio = [base_offset, base_offset + 1, base_offset + 2]
                base_offset += 3
                # block = list(range(base_offset, base_offset + num_views))
                # base_offset += num_views
                if kept[cmd] < target_per_cmd:
                    keep_indices.extend(trio)
                    # keep_indices.extend(block)
                    kept[cmd] += 1
                if len(keep_indices) >= batch_size:
                    break

            X_rgbd  = [X_rgbd[j]  for j in keep_indices]
            X_speed = [X_speed[j] for j in keep_indices]
            X_cmd   = [X_cmd[j]   for j in keep_indices]
            Y_out   = [Y_out[j]   for j in keep_indices]
            Y_speed = [Y_speed[j] for j in keep_indices]

        elif 0 < total < batch_size:
            need = batch_size - total
            center_positions, base_offset = [], 0
            for _ in range(len(batch_entries)):
                center_positions.append(base_offset + 1)
                base_offset += 3
                # center_positions.append(base_offset + center_idx_in_block)
                # base_offset += num_views
            minor_centers = [
                center_positions[bi]
                for bi, (cmd, _) in enumerate(batch_entries)
                if cmd in ("LEFT", "RIGHT", "STRAIGHT")
            ]
            pool = minor_centers if minor_centers else center_positions
            for _ in range(need):
                src = pool[dup_ptr_global % len(pool)]
                dup_ptr_global += 1
                X_rgbd.append(X_rgbd[src]); X_speed.append(X_speed[src]); X_cmd.append(X_cmd[src])
                Y_out.append(Y_out[src]);   Y_speed.append(Y_speed[src])

        # (opsional) acak *semua* item dalam batch (akan mengacak antar-view juga)
        if shuffle_full:
            perm = rng.permutation(len(X_rgbd))
            X_rgbd  = [X_rgbd[k]  for k in perm]
            X_speed = [X_speed[k] for k in perm]
            X_cmd   = [X_cmd[k]   for k in perm]
            Y_out   = [Y_out[k]   for k in perm]
            Y_speed = [Y_speed[k] for k in perm]
        else:
            # kalau hanya mau acak per-triplet SETELAH expand: permutasikan block 3
            if shuffle_triplets:
                n_triplets = len(X_rgbd) // 3
                order = rng.permutation(n_triplets)
            # if shuffle_triplets and num_views > 1:
            #     n_blocks = len(X_rgbd) // num_views
            #     order = rng.permutation(n_blocks)
                perm = []
                for t in order:
                    perm.extend([3*t, 3*t+1, 3*t+2])
                X_rgbd  = [X_rgbd[k]  for k in perm]
                X_speed = [X_speed[k] for k in perm]
                X_cmd   = [X_cmd[k]   for k in perm]
                Y_out   = [Y_out[k]   for k in perm]
                Y_speed = [Y_speed[k] for k in perm]

        yield (
            (np.array(X_rgbd, dtype=np.float32),
             np.array(X_speed, dtype=np.float32),
             np.array(X_cmd, dtype=np.float32)),
            {"c_out": np.array(Y_out, dtype=np.float32),
             "pred_speed":     np.array(Y_speed, dtype=np.float32)}
        )

        # bersih-bersih list
        del batch_entries, X_rgbd, X_speed, X_cmd, Y_out, Y_speed

class ControlSelectorLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        branches, cmd_onehot = inputs
        branches = tf.stack(branches, axis=1)        # shape: (batch, 4, 3)
        cmd_onehot = tf.expand_dims(cmd_onehot, -1)  # shape: (batch, 4, 1)
        return tf.reduce_sum(branches * cmd_onehot, axis=1)

class ControlPostprocess(tf.keras.layers.Layer):
    def call(self, x):
        steer    = tf.tanh(x[:, 0:1])
        throttle = tf.sigmoid(x[:, 1:2])
        brake    = tf.sigmoid(x[:, 2:3])
        return tf.concat([steer, throttle, brake], axis=-1)
    def get_config(self):
        return super().get_config()
    
def act_loss(y_true, y_pred):
    w = tf.constant([0.5, 0.45, 0.05], dtype=y_pred.dtype)  # ✅ Sesuaikan dtype
    diff = tf.abs(y_true - y_pred)
    return tf.reduce_mean(tf.reduce_sum(w * diff, axis=-1))

# === METRICS slice-wise untuk control_output ===
def met_steer(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true[:, 0:1] - y_pred[:, 0:1]))

def met_throt(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true[:, 1:2] - y_pred[:, 1:2]))

def met_brake(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true[:, 2:3] - y_pred[:, 2:3]))

def cil_model():
    from tensorflow.keras import layers, models, regularizers
    from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Conv2D, Flatten, ReLU, Concatenate, Lambda, LayerNormalization

    reg = None

    input_rgbd = Input(shape=(88, 200, 4), name='rgbd_input')
    input_speed = Input(shape=(1,), name='speed_input')
    input_command = Input(shape=(4,), name='command_input')

    def cbam_block(x, reduction_ratio=8, name=None):
        # --- Channel Attention ---
        channel = x.shape[-1]
        avg_pool = tf.reduce_mean(x, axis=[1,2], keepdims=True)
        max_pool = tf.reduce_max(x, axis=[1,2], keepdims=True)
        shared_mlp = tf.keras.Sequential([
            layers.Dense(channel // reduction_ratio, activation='relu', use_bias=False),
            layers.Dense(channel, activation='sigmoid', use_bias=False)
        ])
        ca = shared_mlp(tf.squeeze(avg_pool, [1,2])) + shared_mlp(tf.squeeze(max_pool, [1,2]))
        ca = tf.reshape(tf.nn.sigmoid(ca), [-1,1,1,channel])
        x = x * ca

        # --- Spatial Attention ---
        avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(x, axis=-1, keepdims=True)
        concat = tf.concat([avg_pool, max_pool], axis=-1)
        sa = layers.Conv2D(1, (7,7), padding='same', activation='sigmoid', use_bias=False)(concat)
        x = x * sa
        return x

    x = input_rgbd
    # Conv1: 32, k=5, s=2
    x = Conv2D(32, (5,5), strides=2, padding='same', kernel_regularizer=reg)(x)
    x = BatchNormalization()(x); x = ReLU()(x); x = Dropout(0.2)(x)

    # Conv2: 32, k=3, s=1
    x = Conv2D(32, (3,3), strides=1, padding='same', kernel_regularizer=reg)(x)
    x = BatchNormalization()(x); x = ReLU()(x); x = Dropout(0.2)(x)

    # Conv3: 64, k=3, s=2
    x = Conv2D(64, (3,3), strides=2, padding='same', kernel_regularizer=reg)(x)
    x = BatchNormalization()(x); x = ReLU()(x); x = Dropout(0.2)(x)

    # Conv4: 64, k=3, s=1
    x = Conv2D(64, (3,3), strides=1, padding='same', kernel_regularizer=reg)(x)
    x = BatchNormalization()(x); x = ReLU()(x); x = Dropout(0.2)(x)

    # Conv5: 128, k=3, s=2
    x = Conv2D(128, (3,3), strides=2, padding='same', kernel_regularizer=reg)(x)
    x = BatchNormalization()(x); x = ReLU()(x); x = Dropout(0.2)(x)

    # Conv6: 128, k=3, s=1
    x = Conv2D(128, (3,3), strides=1, padding='same', kernel_regularizer=reg)(x)
    x = BatchNormalization()(x)
    x = cbam_block(x, name='cbam6')   # CBAM sebelum ReLU
    x = ReLU()(x)
    x = Dropout(0.2)(x)               # Dropout setelah ReLU

    # Conv7: 256, k=3, s=2
    x = Conv2D(256, (3,3), strides=2, padding='same', kernel_regularizer=reg)(x)
    x = BatchNormalization()(x); x = ReLU()(x); x = Dropout(0.2)(x)

    # Conv8: 256, k=3, s=1
    # ✅ Final recommended Conv8 block
    x = Conv2D(256, (3,3), strides=1, padding='same', kernel_regularizer=reg)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(0.2)(x)          # Regularisasi sebelum CBAM
    x = cbam_block(x, name='cbam8')  # CBAM jadi filter akhir

    # FC 512 → Dropout 0.5 → FC 512 → Dropout 0.5
    x = Flatten()(x)
    x = Dense(512, activation='relu', kernel_regularizer=reg)(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu', kernel_regularizer=reg)(x)
    x = Dropout(0.5)(x)
    features_p = x  # image features

    s = Dense(128, activation='relu', kernel_regularizer=reg)(input_speed)
    s = Dense(128, activation='relu', kernel_regularizer=reg)(s)
    features_m = s

    # ===== Auxiliary Branch for speed prediction =====
    speed_branch = Dense(128, activation='relu', kernel_regularizer=reg)(features_p)
    speed_branch = Dense(128, activation='relu', kernel_regularizer=reg)(speed_branch)
    speed_branch = Dense(128, activation='relu', kernel_regularizer=reg)(speed_branch)
    pred_speed = Dense(1, activation='tanh', name='pred_speed')(speed_branch)

    joint = Concatenate(name="p_m_concat")([features_p, features_m])
    joint = Dense(256, activation='relu', kernel_regularizer=reg, name="joint_fc")(joint)

    def branch_block(name):
        b = Dense(128, activation='relu', kernel_regularizer=reg)(joint)
        b = Dense(128, activation='relu', kernel_regularizer=reg)(b)
        b = Dense(128, activation='relu', kernel_regularizer=reg)(b)  
        return [
            Dense(1, name=f'{name}_steer')(b),
            Dense(1, name=f'{name}_throttle')(b),
            Dense(1, name=f'{name}_brake')(b)
        ]

    branch_outputs = {k: branch_block(k) for k in ['lanefollow', 'left', 'right', 'straight']}

    def control_selector(branch_outputs, command_input):
        branches_concat = [tf.concat(branch_outputs[k], axis=-1) for k in ['lanefollow', 'left', 'right', 'straight']]
        return ControlSelectorLayer(name='control_selector')([branches_concat, command_input])

    control_out = control_selector(branch_outputs, input_command)
    control_output = ControlPostprocess(name='c_out')(control_out)

    model = models.Model(
        inputs=[input_rgbd, input_speed, input_command],
        outputs=[control_output, pred_speed]
    )
    return model

class EpochLRLogger(tf.keras.callbacks.Callback):
    def _base_opt(self):
        opt = self.model.optimizer
        return opt._optimizer if isinstance(opt, tf.keras.mixed_precision.LossScaleOptimizer) else opt

    def on_epoch_end(self, epoch, logs=None):
        opt = self._base_opt()
        # panggil sekali per-epoch → sync tapi amat jarang, tidak mengganggu throughput
        if callable(opt.learning_rate):
            lr_t = opt.learning_rate(opt.iterations)  # tf.Tensor
        else:
            lr_t = opt.learning_rate
        lr = float(tf.keras.backend.get_value(lr_t))
        print(f"[LR][Epoch {epoch+1}] {lr:.8f}")

class EpochCSVLogger(tf.keras.callbacks.Callback):
    """
    Menulis log per-epoch ke CSV, dengan kolom:
    - iteration : total global iterations (optimizer.iterations)
    - ... + semua metrik dari logs (loss, dll.)
    """
    def __init__(self, filename):
        super().__init__()
        self.filename = filename
        self.append   = os.path.exists(filename)
        self.file     = None
        self.writer   = None
        self.keys     = None

    def _base_opt(self):
        opt = self.model.optimizer
        if isinstance(opt, tf.keras.mixed_precision.LossScaleOptimizer):
            opt = opt._optimizer
        return opt

    def on_train_begin(self, logs=None):
        self.file = open(self.filename, 'a' if self.append else 'w',
                         newline='', buffering=1024*1024)
        self.writer = None
        self.keys   = None

    def on_epoch_end(self, epoch, logs=None):
        logs = dict(logs or {})

        # iteration = global counter dari optimizer
        logs['iteration'] = int(tf.keras.backend.get_value(self._base_opt().iterations))

        if self.writer is None:
            self.keys = sorted(logs.keys())
            self.writer = csv.DictWriter(self.file, fieldnames=self.keys)
            if not self.append:
                self.writer.writeheader()

        self.writer.writerow({k: logs.get(k, "") for k in self.keys})
        self.file.flush()
        print(f"[CSV] flushed at iteration={logs['iteration']} → {self.filename}")

    def on_train_end(self, logs=None):
        if self.file:
            self.file.flush()
            self.file.close()
            self.file = None

def log_memory(msg):
    process = psutil.Process()
    rss = process.memory_info().rss / 1024 / 1024
    vmem = psutil.virtual_memory()
    available = vmem.available / 1024 / 1024
    free = vmem.free / 1024 / 1024
    print(f"[RAM MEMORY] {msg}: RSS={rss:.2f} MB | Free={free:.2f} MB | Available={available:.2f} MB")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', type=int, default=1)
    args = parser.parse_args()

    output_base_dir = f"mul_cil_a2_run{args.run_id}"
    os.makedirs(output_base_dir, exist_ok=True)

    # ===== Milestone saving config =====
    MILESTONES = [100_000, 200_000, 300_000, 400_000, 500_000]

    def _milestone_path(base_dir, step):
        k = step // 1000
        return os.path.join(base_dir, f"saved_model_at_{k}k.h5")

    # Catat milestone yang sudah tersimpan (jika melanjutkan training)
    already_saved_milestones = set()
    for ms in MILESTONES:
        if os.path.exists(_milestone_path(output_base_dir, ms)):
            already_saved_milestones.add(ms)

    print("Policy:", mixed_precision.global_policy().name)

    model_path_latest = os.path.join(output_base_dir, "saved_model_latest.h5")
    progress_file = os.path.join(output_base_dir, "training_progress.txt")
    interval = 500
    total_files = len(h5_files) // interval
    total_files = total_files * interval
    print(f"[INFO] Total file yang akan dipakai: {total_files}")

    # total_iteration = 1_000_000    
    total_iteration = 500_000      
    # total_iteration = 300_000    
    # total_iteration = 100_000
    lr_iteration = 50_000
    save_model_at = 100_000
    
    # ===== Resume checkpoint =====
    resume_step = 0
    resume_chunk_idx = 0
    if os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            for line in f:
                if line.startswith("step="):
                    resume_step = int(line.strip().split("=")[1])
                elif line.startswith("chunk_start_idx="):
                    resume_chunk_idx = int(line.strip().split("=")[1])

    # ===== Load model or build new =====
    if os.path.exists(model_path_latest):
        print(f"[INFO] Loading model from checkpoint: {model_path_latest}")
        model = tf.keras.models.load_model(
            model_path_latest,
            compile=False,
            custom_objects={
                'act_loss': act_loss,
                'ControlSelectorLayer': ControlSelectorLayer,
                'ControlPostprocess':ControlPostprocess,
            }
        )

        # Buat ulang optimizer dengan learning rate schedule
        lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries=[lr_iteration * i for i in range(1, 10)],
            values=[0.0002 / (2 ** i) for i in range(10)]
        )
        optimizer = Adam(learning_rate=lr_schedule)

        # Kompilasi model dengan optimizer baru
        model.compile(
            optimizer=optimizer,
            loss={
                'c_out': act_loss,
                'pred_speed': 'mae'
            },
            loss_weights={
                'c_out': 0.95,
                'pred_speed': 0.05
            },
            metrics={
                'c_out': [met_steer, met_throt, met_brake],
                # 'pred_speed'    : [tf.keras.metrics.MeanAbsoluteError(name='mae')],
            }
        )

        # ✅ Setelah compile, atur ulang step optimizer
        if resume_step > 0:
            optimizer.iterations.assign(resume_step)
    else:
        print("[INFO] Building new model")
        model = cil_model()

        lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries=[lr_iteration * i for i in range(1, 10)],
            values=[0.0002 / (2 ** i) for i in range(10)]
        )
        optimizer = Adam(learning_rate=lr_schedule)
        model.compile(
            optimizer=optimizer,
            loss={
                'c_out': act_loss,
                'pred_speed': 'mae'
            },
            loss_weights={
                'c_out': 0.95,
                'pred_speed': 0.05
            },
            metrics={
                'c_out': [met_steer, met_throt, met_brake],
                # 'pred_speed'    : [tf.keras.metrics.MeanAbsoluteError(name='mae')],
            }
        )

    print("Optimizer type:", type(model.optimizer))

    while resume_step < total_iteration:
        for start_idx in range(resume_chunk_idx, total_files, interval):
            end_idx = min(start_idx + interval, total_files)
            chunk_name = f"chunk_{start_idx:05d}_{end_idx:05d}"

            print("=" * 60)
            print(f"[INFO] Training chunk {chunk_name}, Iteration={resume_step}")
            print("=" * 60)

            gc.collect()

            selected_h5_files = h5_files[start_idx:end_idx]
            opened_files, index_per_cmd = load_and_index_h5_files(selected_h5_files)

            ds = (tf.data.Dataset.from_generator(
                lambda: simple_balanced_tf_generator(
                    batch_size=batch_size,
                    index_per_cmd=index_per_cmd,
                    opened_files=opened_files,
                    shuffle_triplets=True,   # atau False, sesuai kebutuhan
                    shuffle_full=True,      # atau True
                    seed=resume_step + start_idx  # agar berubah antar chunk/step
                ),
                output_signature=(
                    (
                        tf.TensorSpec(shape=(None, 88, 200, 4), dtype=tf.float32),
                        tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
                        tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
                    ),
                    {
                        "c_out": tf.TensorSpec(shape=(None, 3), dtype=tf.float32),
                        "pred_speed":     tf.TensorSpec(shape=(None, 1), dtype=tf.float32)
                    }
                )
            )
            # buffer kecil saja karena elemen=1 batch
            .shuffle(buffer_size=16, reshuffle_each_iteration=True)
            .prefetch(tf.data.AUTOTUNE))

            steps_per_epoch = len(index_per_cmd["LANEFOLLOW"]) // (batch_size // (4*3))
            # steps_per_epoch = len(index_per_cmd["LANEFOLLOW"]) // (batch_size // (4))
            print(f"[INFO] {chunk_name} - Steps: {steps_per_epoch} - LANEFOLLOW count: {len(index_per_cmd['LANEFOLLOW'])}")

            callbacks = [
                EpochCSVLogger(os.path.join(output_base_dir, 'training_log.csv')),
                EpochLRLogger(),
            ]

            model.fit(
                ds,
                steps_per_epoch=steps_per_epoch,
                epochs=1,
                callbacks=callbacks,
                verbose=1
            )

            prev_step = resume_step
            resume_step += steps_per_epoch

            # Simpan step dan chunk index setelah selesai fit
            with open(progress_file, "w") as f:
                f.write(f"step={optimizer.iterations.numpy()}\n")
                f.write(f"chunk_start_idx={start_idx + interval}\n")

            del ds
            for f in opened_files:
                f.close()
            del opened_files
            del index_per_cmd
            gc.collect()

            model.save(model_path_latest)
            log_memory("After saving and deleting model")

            # ==== CEK & SIMPAN MILESTONE ====
            # Jika dalam chunk ini kita melewati 1 atau lebih milestone, simpan untuk masing-masing.
            for ms in MILESTONES:
                if (ms not in already_saved_milestones) and (prev_step < ms <= resume_step):
                    ms_path = _milestone_path(output_base_dir, ms)
                    model.save(ms_path)
                    already_saved_milestones.add(ms)
                    print(f"[MILESTONE] Reached {ms:,} steps → saved: {ms_path}")

            if resume_step >= total_iteration:
                break

    # ===== Cleanup final =====
    del model
    del optimizer
    K.clear_session()
    gc.collect()
    log_memory("Final cleanup")
    print(f"sudah mencapai {total_iteration}")   
