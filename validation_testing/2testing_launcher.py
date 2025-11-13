# Copyright (c) 2025 Endang Rusiana.
# This work is licensed under the terms of the MIT License.
# For a copy, see <https://opensource.org/licenses/MIT>.

import subprocess
import time
import os
import signal
import socket
import sys
from pathlib import Path

# === Fungsi utilitas ===
def is_port_open(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex((host, port)) == 0

def kill_carla():
    print("[KILL] 🔪 Menghentikan proses CARLA...")
    os.system("pkill -9 -f CarlaUE4")
    time.sleep(2)
    for _ in range(15):
        if not is_port_open("127.0.0.1", 2000):
            print("[OK] ✅ Port 2000 sudah bebas.")
            return
        print("[WAIT] Port 2000 masih digunakan... tunggu 2 detik")
        time.sleep(2)
    print("[ERROR] ❌ Port 2000 tidak pernah bebas. Keluar.")

# === Konfigurasi path ===
CARLA_PATH = "/opt/carla-simulator"
CARLA_EXECUTABLE = os.path.join(CARLA_PATH, "CarlaUE4.sh")
EVALUATION_SCRIPT = os.path.join(CARLA_PATH, "PythonAPI/examples/3testing_model.py")
PYTHON_EXEC = "/home/edgpc/anaconda3/envs/cil_tf37/bin/python"  # ganti sesuai path environment cil_tf37

print(f"[INFO] Menjalankan: {EVALUATION_SCRIPT}")
# === Simpan PID CARLA global untuk digunakan saat kill ===
carla_process = None

def main():
    global carla_process
    print("▶️ Menjalankan CARLA simulator dalam mode offscreen...")
    carla_process = subprocess.Popen(
        [CARLA_EXECUTABLE, "-RenderOffScreen"],
        cwd=CARLA_PATH,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid
    )
    print("⏳ Menunggu CARLA agar stabil...")
    time.sleep(10)

    eval_dir = Path(EVALUATION_SCRIPT).parent

    model_arch_selected = [
         "mul_cil_a2",
    ]

    model_run_selected = [
         1,
    ]

    model_saved_selected = [
         "500k",
    ]

    TASK_TRIALS = {
        "straight": 6,
        "one_turn": 6,
        "navigation": 6,
        "nav_dynamic": 7,
    }
    
    # Kombinasi flag yang ingin dijalankan (urutannya dipertahankan)
    TESTING_MODE = [
        "training_conditions",
        "new_town",
        "new_weather",
        "new_town_weather",
    ]

    for trial_count in range(1, 4):
    # for trial_count in range(1, 2):
        for index, model_count in enumerate(model_arch_selected):
            for test_count in TESTING_MODE:
                if test_count == "training_conditions":
                    weather_list = ["ClearNoon", "ClearSunset", "WetNoon", "HardRainNoon"]
                    town_selected = "Town01"
                elif test_count == "new_town":
                    weather_list = ["ClearNoon", "ClearSunset", "WetNoon", "HardRainNoon"]
                    town_selected = "Town02"
                elif test_count == "new_weather":
                    weather_list = ["WetSunset", "SoftRainSunset"]
                    town_selected = "Town01"
                elif test_count == "new_town_weather":
                    weather_list = ["WetSunset", "SoftRainSunset"]
                    town_selected = "Town02"
                else:
                    continue  # safety

                for weather_count in weather_list:
                    for task, num_trial in TASK_TRIALS.items():
                        for i in range(1, num_trial + 1):
                            print(f"=== TRIAL KE- {i} ===")
                            print(f"🚗 Menjalankan evaluasi {model_count} | {model_run_selected[index]} | {model_saved_selected[index]}")
                            print(f"Trial Mode:{trial_count} | Test:{test_count} | Town:{town_selected} | Weather:{weather_count} | {task}:{i}")

                            # ==== Restart CARLA sebelum setiap trial ====
                            kill_carla()
                            print("[RESTART] 🔄 Menjalankan ulang CARLA...")
                            time.sleep(10)
                            carla_process = subprocess.Popen(
                                [CARLA_EXECUTABLE, "-RenderOffScreen"],
                                cwd=CARLA_PATH,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                                preexec_fn=os.setsid
                            )
                            print("⏳ Menunggu CARLA agar stabil...")
                            time.sleep(10)
                            # ============================================

                            cmd = [
                                PYTHON_EXEC, EVALUATION_SCRIPT,
                                f"--task={task}",
                                "--model_arch", model_count,
                                "--run", str(model_run_selected[index]),
                                "--model_saved", model_saved_selected[index],
                                "--trial_per_mode", str(trial_count),
                                "--testing_mode", test_count,
                                "--weather", weather_count,
                                "--town", town_selected,
                            ]
                            try:
                                subprocess.run(cmd, cwd=eval_dir, check=True)
                            except subprocess.CalledProcessError as e:
                                print(f"❌ Gagal pada task={task}, testing_mode={test_count}, trial={i}. Kode keluar: {e.returncode}")
                            time.sleep(5)

# === Tangani Ctrl+C (SIGINT) atau kill (SIGTERM) ===
def signal_handler(sig, frame):
    print("\n[INTERRUPT] ❗ Dihentikan oleh user atau sistem.")
    kill_carla()
    sys.exit(1)

# Register signal handler
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

if __name__ == "__main__":
    try:
        main()
    finally:
        kill_carla()
        print("✅ Evaluasi selesai dan CARLA simulator dimatikan.")
