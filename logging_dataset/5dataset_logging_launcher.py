# Copyright (c) 2025 Endang Rusiana.
# This work is licensed under the terms of the MIT License.
# For a copy, see <https://opensource.org/licenses/MIT>.

import subprocess
import time
import psutil
import signal
import os
import socket
import sys

CARLA_COMMAND = ["/opt/carla-simulator/CarlaUE4.sh", "-RenderOffScreen"]
PYTHON_COMMAND = [
    "/home/edgpc/anaconda3/envs/carla_env/bin/python",
    "/opt/carla-simulator/PythonAPI/examples/6dataset_logging.py"
]

# Kombinasi 1 siklus (weather, lead_vehicle_flag)
COMBOS = [
# ("ClearNoon",     False, False),
# ("ClearNoon",     False, True),
# ("ClearNoon",     True, False)

#========================================= 1 follow car in row ======================================================
    ("ClearNoon",    False, False),
    ("ClearSunset",  True,  False),
    ("WetNoon",      False, True),
    ("HardRainNoon", True,  False),

    ("ClearNoon",    True,  False),
    ("ClearSunset",  False, True),
    ("WetNoon",      False, False),
    ("HardRainNoon", False, True),

    ("ClearNoon",    False, True),
    ("ClearSunset",  False, False),
    ("WetNoon",      True,  False),
    ("HardRainNoon", False, False),
]

# Atur jumlah siklus (None = tak terbatas / ulangi selamanya)
MAX_CYCLES = None  # contoh: 3 untuk 3 kali putaran, atau None untuk infinite

def is_carla_running():
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmd = proc.info.get('cmdline') or []
            if any("CarlaUE4" in s for s in cmd):
                return proc
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None

def is_port_open(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        return sock.connect_ex((host, port)) == 0

def kill_carla():
    print("Wait 5 sec")
    time.sleep(5)
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
    print("Wait 5 sec")
    time.sleep(5)

def launch_carla():
    print("[SPAWN] 🚗 Starting CARLA...")
    subprocess.Popen(CARLA_COMMAND)
    time.sleep(10)

def run_get_training_data(weather, lead_vehicle=False, add_traffic=False):
    cmd = PYTHON_COMMAND + ["--weather", weather, 
                            # "--headless",
                            ]
    if lead_vehicle:
        cmd.append("--lead_vehicle")    
    if add_traffic:
        cmd.append("--add_traffic")
    lead_str = " & --lead_vehicle" if lead_vehicle else ""
    traffic_str = " & --add_traffic" if add_traffic else ""
    print(f"[RUN] ▶️ Menjalankan logging dengan weather={weather}{lead_str}")
    return subprocess.run(cmd)

def signal_handler(sig, frame):
    print("\n[INTERRUPT] ❗ Dihentikan oleh user atau sistem.")
    kill_carla()
    sys.exit(1)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def one_cycle():
    """Jalankan tepat 1 putaran atas semua COMBOS."""
    for i, (weather, lead_flag, traffic_flag) in enumerate(COMBOS, start=1):
        print(f"\n===== 🚧 ({i}/{len(COMBOS)}) MULAI LOGGING: weather={weather}, lead={lead_flag} =====, traffic={traffic_flag}")

        if is_carla_running():
            kill_carla()
        launch_carla()

        result = run_get_training_data(weather, lead_flag, traffic_flag)

        if result.returncode == 0:
            print(f"[DONE] ✅ Logging selesai untuk {weather} (lead={lead_flag}) (traffic={traffic_flag})")
            # 👉 langsung hentikan proses di sini
            kill_carla()
            sys.exit(0)
        else:
            print(f"[RETRY→NEXT] 🔁 Logging gagal untuk {weather} (lead={lead_flag}) (traffic={traffic_flag}). Lanjut ke kombinasi berikutnya.")

def main():
    cycle_idx = 0
    try:
        while True:
            cycle_idx += 1
            print(f"\n\n========== 🔁 MULAI SIKLUS #{cycle_idx} (total COMBOS: {len(COMBOS)}) ==========")
            one_cycle()
            print(f"========== ✅ SIKLUS #{cycle_idx} SELESAI ==========\n")

            # Jika dibatasi jumlah siklus, cek dan break
            if MAX_CYCLES is not None and cycle_idx >= MAX_CYCLES:
                break

            # (Opsional) jeda antar siklus
            time.sleep(3)

    finally:
        kill_carla()
        print("✅ Semua siklus selesai atau dihentikan. CARLA simulator dimatikan.")

if __name__ == "__main__":
    main()
