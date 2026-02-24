# 🚗 Autonomous Driving — MCILAE  
### Multimodal Conditional Imitation Learning with Attention & Error Compensation  
**CARLA Simulator 0.9.15 (UE4.26)**  
**Author: Endang Rusiana**

---

## 📌 Overview
This repository contains the complete implementation of **MCILAE**, an enhanced multimodal end-to-end autonomous driving framework that extends CIL and MCIL through:

- **RGB–Depth multimodal fusion**
- **CBAM attention refinement (applied after Conv6 & Conv8)**
- **Feedback-based throttle/brake correction (Error Compensation)**
- **Throttle-phase-aware dataset logging (Acceleration, Cruising, Deceleration)**
- Robust evaluation across multiple towns, weathers, and dynamic traffic in CARLA

Developed using **CARLA 0.9.15**, **Python 3.7**, **TensorFlow 2.10/2.11**, and **RTX 3060 GPU**.

---

## 📁 Repository Structure
```
autonomous_driving_mcilae/
├── logging_dataset/
│   ├── 5dataset_logging_launcher.py
│   └── 6dataset_logging.py
├── training/
│   ├── 0_train_launcher.py
│   └── 8_mul_cil_a2.py
├── validation_testing/
│   ├── 0validation_launcher.py
│   ├── 1validation_model.py
│   ├── 2testing_launcher.py
│   ├── 3testing_model.py
│   └── 4test_weather.py
├── requirements.txt
└── README.md
```

---

## 🔧 System Requirements
| Component | Version |
|----------|---------|
| CARLA | **0.9.15 (UE4.26)** |
| CUDA Toolkit | **12.8** |
| NVIDIA Driver | Compatible with CUDA ≥ 12.8 |
| Python | **3.7** (conda env: `cil_tf37`) |
| TensorFlow | **2.10 / 2.11** |
| GPU | RTX 3060 12GB |
| CPU | Intel i7-12700F |
| RAM | 32GB DDR4 |
| OS | Ubuntu 20.04 |

---

## 📦 Installation

### 1. Clone repository
```bash
git clone https://github.com/endangrusiana123/autonomous_driving_mcilae.git
cd autonomous_driving_mcilae
```

### 2. Create conda environment
```bash
conda create -n cil_tf37 python=3.7
conda activate cil_tf37
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Running the System

### 1. Launch CARLA 0.9.15
```bash
./CarlaUE4.sh
```

---

## 🧪 Dataset Logging
```bash
python logging_dataset/5dataset_logging_launcher.py
```
## 📁 Sample Dataset
A sample 200MB HDF5 dataset is available for download:

👉 [Download Sample Dataset](https://drive.google.com/drive/folders/1CDMBr3BafIZR_o4RFifhutbemgIPxX4x?usp=sharing)

## 📝 Output Dataset Logging

Each dataset file (`cil_batch_XXXXX.h5`) contains **200 frames** with the following structure:

### **1. RGB Images (3 views)**
- Dataset: `rgb`
- Shape: **(200, 3, 88, 200, 3)**
- Views: left, center, right (cropped from 330×120 → 88×200)

### **2. Depth Images (3 views)**
- Dataset: `depth`
- Shape: **(200, 3, 88, 200)**
- 8-bit depth after decoding + clipping + median filter

### **3. LiDAR BEV (2-layer)**
- Datasets: `lidar_above`, `lidar_ground`
- Shape: **(200, 88, 198)**
- Generated from ray-cast LiDAR → BEV histogram

### **4. Metadata (per-column datasets)**  
Each metadata field is written as **its own dataset**:

| Dataset | Description |
|--------|-------------|
| `frame_id` | Global frame index |
| `speed` | Speed (km/h) |
| `road_option` | LEFT / RIGHT / STRAIGHT / LANEFOLLOW |
| `steer` | Controller steer value |
| `steer_noise` | Injected Gaussian noise |
| `steer_resultant` | Final applied steer |
| `throttle` | Throttle value |
| `brake` | Brake value |
| `red_light` | Red light flag |
| `at_traffic_light` | Traffic-light proximity flag |
| `weather` | Current weather (string) |
| `vehicle_in_front` | 1 if vehicle ahead |
| `is_curve` | 1 if on curve |
| `is_junction` | 1 if entering junction |
| `speed_kmh_t` | Speed(t) |
| `speed_kmh_t_1` | Speed(t-1) |
| `speed_kmh_t_2` | Speed(t-2) |
| `speed_kmh_t_3` | Speed(t-3) |

### **5. Local Waypoints (relative hero frame)**
Waypoints are stored as:

```
wp0_x, wp0_y,
wp1_x, wp1_y,
wp2_x, wp2_y,
wp3_x, wp3_y,
wp4_x, wp4_y
```

---

Total metadata columns: **27 datasets**  
Total sensors per frame: **RGB, Depth, LiDAR BEV, Metadata, Waypoints**  

## 🎓 Training MCILA/MCILAE
```bash
python training/0_train_launcher.py
```

Supports:
- Mixed precision  
- Balanced batch sampling (per command)  
- Checkpoint saving  
- Multi-file HDF5 dataset streaming  

---

## ✔ Validation
```bash
python validation_testing/0validation_launcher.py
```

---

## 🏁 Testing (Navigation, Dynamic, Obstacle)
```bash
python validation_testing/2testing_launcher.py
```

---

## 🎥 Project Demonstration
[![Watch the video](https://img.youtube.com/vi/zJhk1q0ixiY/0.jpg)](https://www.youtube.com/watch?v=zJhk1q0ixiY)

---

## 📄 Master’s Thesis
Official thesis repository:
https://openlibrary.telkomuniversity.ac.id/home/catalog/id/245798/

---

## 📄 Citation
```
@misc{rusiana2025mcilae,
  author       = {Endang Rusiana},
  title        = {MCILAE: Multimodal Conditional Imitation Learning with Attention and Error Compensation for Autonomous Driving},
  howpublished = {\url{https://github.com/endangrusiana123/autonomous_driving_mcilae}},
  institution  = {Telkom University},
  note         = {Thesis available at \url{https://openlibrary.telkomuniversity.ac.id/home/catalog/id/245798/}},
  year         = {2026}
}
```

---

## 📜 License
This project is released under the **MIT License**, fully compatible with the CARLA example code license.

---

## 🤝 Acknowledgements
This project was developed as part of a Master's research at Telkom University.  
The author thanks the developers and researchers whose public work made this implementation possible, including:

- CARLA Simulator (open-source)  
- TensorFlow and Keras  
- Prior research on Conditional Imitation Learning (Codevilla et al.)  
- Multimodal CIL (Xiao et al.)  
- CBAM Attention (Woo et al.)  

All referenced work is cited and used only as academic foundations.  

