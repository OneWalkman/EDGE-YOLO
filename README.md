# EdgeLine-YOLO: Real-Time Industrial Defect Detection  
*A modular YOLO-based framework for high-precision, real-time industrial visual inspection.*  
:contentReference[oaicite:0]{index=0}

---

## Table of Contents
- [Project Overview](#project-overview)
- [Core Features](#core-features)
- [Repository Structure](#repository-structure)
- [License](#license)

---

## Project Overview
Industrial defect detection is critical to manufacturing quality control, where defects are typically **small**, **low-contrast**, and embedded in **complex texture patterns**. Real-world deployments also impose strict **real-time** and **edge-device** constraints.

EdgeLine-YOLO addresses these challenges with a modular enhancement of the YOLO detection framework, integrating:  
- **Long-range context modeling** (Linear Attention)  
- **Frequency-domain representation** (Wavelet-enhanced Neck)  
- **Quality-aligned scoring** (GFLv2 × UniHead)  
:contentReference[oaicite:1]{index=1}

### Key Results (Summary)
- **+2.2% mAP@0.5:0.95** over baseline YOLO11  
- **–1.9% FNR** (fewer missed defects)  
- **Real-time latency preserved (~8 ms on RTX 4090 FP16)**  
- Validated across **GC10-DET**, **DeepPCB**, **Magnetic Tile**, **TILDA Textile**  

---

## Core Features
- **C2PSA Linear Attention Backbone**  
  - Efficient linear attention at S16 stage  
  - Strengthens long-range global reasoning  

- **DSC3K2_Wavelet Neck**  
  - Single-level 2D-DWT sub-band enhancement  
  - Improves edge, crack, and texture sensitivity  

- **GFLv2 × UniHead Detection Head**  
  - Distribution-aware regression (DFL)  
  - Quality Focal Loss for confidence–IoU alignment  
  - More stable NMS ranking in repetitive textures  

- **Unified Experimental Protocol**  
  - Standardized preprocessing, augmentation, and evaluation  
  - Reproducible across datasets and hardware  

- **Edge Deployment–friendly**  
  - Minimal added latency  
  - Maintains shape consistency and plug-and-play modularity  

---

### Algorithm & Model Design
- Designed the **EdgeLine-YOLO** architecture integrating attention, wavelets, and quality-aware detection.  
- Implemented **C2PSA Linear-Attention** module customized for YOLO11 backbone.  
- Built **DSC3K2_Wavelet** fusion module with DWT-based sub-band processing.  
- Replaced YOLO head with **GFLv2 × UniHead** achieving score–quality alignment.

### Innovation & Technical Challenges
- Integrated **linear attention, wavelet transforms, and distributional regression** in a fully modular YOLO-based detector.  
- Achieved **real-time performance** despite multiple enhancements.  
- Addressed **small/low-contrast defect detection** and **threshold instability** in industrial deployments.

---

## Repository Structure
Refer to the Ultralytics repository​ at https://github.com/ultralytics/ultralytics

## License
No license

## Acknowledgement
This repository is based on my Master thesis conducted at Instituto Superior Técnico (IST), University of Lisbon:  
*"EdgeLine-YOLO: Real-Time Industrial Defect Detection"*.

