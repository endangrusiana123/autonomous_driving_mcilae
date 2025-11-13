# Copyright (c) 2025 Endang Rusiana.
# This work is licensed under the terms of the MIT License.
# For a copy, see <https://opensource.org/licenses/MIT>.

import subprocess
import sys
import time

# Gunakan path absolut ke interpreter environment aktif
interpreter = sys.executable  # akan mengarah ke /home/edgpc/anaconda3/envs/carla_env/bin/python

for run_id in range(1, 6):  # dari 1 sampai 5 (range stop tidak inklusif)
    print(f"▶️ mul_cil_a2 training run ke-{run_id}...")
    subprocess.run([interpreter, "-u", "8_mul_cil_a2.py", "--run_id", str(run_id)], check=True)
    time.sleep(2)

    # print(f"▶️ mul_cil training run ke-{run_id}...")
    # subprocess.run([interpreter, "-u", "2_mul_cil.py", "--run_id", str(run_id)], check=True)
    # time.sleep(2)

    # print(f"▶️ cil training run ke-{run_id}...")
    # subprocess.run([interpreter, "-u", "3_cil.py", "--run_id", str(run_id)], check=True)
    # time.sleep(2)

print(f"✅ Selesai run ke-{run_id}\n")

