# Autonomous Driving MCILAE  
### Multimodal Conditional Imitation Learning with Attention & Error Compensation  
*Implementation for CARLA 0.9.15 (UE4.26)*

---

## 📌 Overview  
This repository contains the complete source code used for developing and evaluating  
**Multimodal Conditional Imitation Learning with Attention and Error Compensation (MCILAE)**  
for end-to-end autonomous driving in the CARLA simulator.

The project extends the baseline CIL and MCIL architectures by integrating:

- **Multimodal RGB–Depth fusion**
- **Attention refinement (CBAM)**
- **Longitudinal feedback correction (Error Compensation)**
- **Throttle-phase-aware dataset design**
- **Robust evaluation under multiple towns, weathers, and dynamic traffic**

This implementation supports full dataset logging, training, validation, testing,  
and visualization inside CARLA 0.9.15.

---

## 📁 Repository Structure
autonomous_driving_mcilae/
├── logging_dataset/
│ ├── 5dataset_logging_launcher.py
│ └── 6dataset_logging.py
├── training/
│ ├── 0_train_launcher.py
│ └── 8_mul_cil_a2.py
├── validation_testing/
│ ├── 0validation_launcher.py
│ ├── 1validation_model.py
│ ├── 2testing_launcher.py
│ ├── 3testing_model.py
│ └── 4test_weather.py
├── requirements.txt
└── README.md
