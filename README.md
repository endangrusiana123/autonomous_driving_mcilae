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
./CarlaUE4.sh -quality-level=Low
```

---

## 🧪 Dataset Logging
```bash
python logging_dataset/5dataset_logging_launcher.py
```

Outputs:
- RGB camera  
- Depth camera  
- Speed  
- Steering / throttle / brake  
- High-level command (LEFT, RIGHT, STRAIGHT, LANEFOLLOW)  
- Weather metadata  
- Traffic metadata  

---

## 🎓 Training MCIL/MCILA/MCILE/MCILAE
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

## 🌦 Weather Stress Test
```bash
python validation_testing/4test_weather.py
```

Press **W** to switch weather presets.

---

## 📄 Citation
```
@misc{rusiana2025mcilae,
  author       = {Endang Rusiana},
  title        = {MCILAE: Multimodal CIL with Attention and Error Compensation for Autonomous Driving},
  howpublished = {\url{https://github.com/endangrusiana123/autonomous_driving_mcilae}},
  year         = {2025}
}
```

---

## 📜 License
This project is released under the **MIT License**, fully compatible with the CARLA example code license.

---

## 🤝 Acknowledgements
- CARLA Simulator Team  
- Codevilla et al. (CIL)  
- Xiao et al. (MCIL)  
- Woo et al. (CBAM)  
- TensorFlow Research Community  
- Telkom University — Electrical Engineering Graduate Program  
