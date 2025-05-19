# Ultrasound Image Segmentation of Histotripsy Ablation Using DeepLabV3 with Combined Dice-Focal Loss

<img src="https://labs.iitgn.ac.in/muselaboratory/wp-content/uploads/2022/02/TB4x.png" width="200"/>

This repository contains the code and resources for the research project on segmenting histotripsy ablation zones in ultrasound images using DeepLabV3 (and other experimental models) with a combined Dice-Focal loss function.

**Authors:** Shreyans Jain, Soham Gaonkar, Aditya Borate, Mihir Agarwal, Muskan Singh, Prof. Kenneth B. Bader, Prof. Himanshu Shekhar  
**Contact:** shreyans.jain@iitgn.ac.in, soham.gaonkar@iitgn.ac.in, aditya.borate@iitgn.ac.in, agarwalmihir@iitgn.ac.in, himanshu.shekhar@iitgn.ac.in

---

## Table of Contents
1.  [Introduction](#1-introduction)
2.  [Motivation](#2-motivation)
3.  [Methodology](#3-methodology)
    *   [Model Architecture](#model-architecture)
    *   [Loss Function](#loss-function)
4.  [Repository Structure](#4-repository-structure)
5.  [Experimental Setup](#5-experimental-setup)
    *   [Dataset](#dataset)
6.  [Results](#6-results)
7.  [Challenges Faced](#7-challenges-faced)
8.  [Future Work](#8-future-work)
9.  [Citation](#9-citation)
10. [Acknowledgments](#10-acknowledgments)
11. [License](#11-license)

---

## 1. Introduction
Histotripsy is an emerging non-invasive ultrasound-based therapy that uses high-amplitude focused ultrasound pulses to mechanically ablate tissue without heat or incisions. Real-time monitoring and assessment of the ablated region are crucial for effective treatment. This project explores deep learning-based segmentation approaches, primarily focusing on DeepLabV3 with a combined Dice-Focal loss, to accurately identify and segment histotripsy-induced ablation zones from B-mode ultrasound images.

## 2. Motivation
*   Current imaging modalities like MRI/CT for monitoring histotripsy ablation are slow, costly, and not ideal for real-time feedback.
*   Ultrasound imaging is a readily available, cost-effective, and real-time alternative.
*   Accurate segmentation of the ablated region in ultrasound images is challenging due to low contrast, speckle noise, sparse bubble regions (especially in early-pulse frames), and variability in bubble cloud appearance.
*   **Our Goal:** To enable real-time assessment of histotripsy ablation using ultrasound imaging alone by developing robust segmentation models.

## 3. Methodology

The overall workflow involves:
Data Acquisition → Data Annotation → Preprocessing & Augmentation → Supervised Learning → Evaluation.

### Model Architecture
*   The primary model discussed in our poster is **DeepLabV3** with a **ResNet101** backbone, trained from scratch in a fully supervised setting (`code_files/model/deeplabv3_torchvision.py`, `code_files/model/deeplabv3p.py`).
*   DeepLabV3 utilizes atrous (dilated) convolutions and Atrous Spatial Pyramid Pooling (ASPP) to capture multi-scale context effectively.
*   The repository also includes implementations or experiments with other architectures such as U-Net++ (`unetpp.py`), FPN (`fpn.py`), Mask R-CNN (`maskrcnn.py`), and a ResNet18-based CNN (`resnet18cnn.py`), found in the `code_files/model/` directory.

### Loss Function
To handle class imbalance and enhance boundary precision, we primarily employ a combined **Dice + Focal Loss** (`code_files/loss/dice_focal.py`):
`L_Total = λ_Dice * L_Dice + λ_Focal * L_Focal`

*   **Dice Loss:** Maximizes overlap in sparse masks (`code_files/loss/dice.py`).
    `DiceLoss(y, p̂) = 1 - (2p̂y + ε) / (p̂ + y + ε)`
*   **Focal Loss:** Emphasizes hard-to-classify pixels.
    `FocalLoss(p_t) = -α_t (1 - p_t)^γ log(p_t)`
*   Other experimental loss functions like Asymmetric Tversky Loss (`asymmetric_tversky.py`) and a Dice-Focal variant with pulse prior (`dice_focal_pulse_prior.py`) are also available in the `code_files/loss/` directory.

## 4. Repository Structure

```
adi776borate-bubblesegmentation/
├── README.md                      
├── code_files/                     # Main Python scripts and modules
│   ├── config.py                   
│   ├── dataloader.py               
│   ├── latest_copy.ipynb           
│   ├── metric.py                   
│   ├── run_all.py                  
│   ├── test.py                     
│   ├── train.py                    
│   ├── utils.py                    
│   ├── zcleanup.py                 
│   ├── loss/                       
│   │   ├── __init__.py
│   │   ├── asymmetric_tversky.py
│   │   ├── dice.py
│   │   ├── dice_focal.py
│   │   └── dice_focal_pulse_prior.py
│   └── model/                      # Model architecture implementations
│       ├── __init__.py
│       ├── deeplabv3_torchvision.py
│       ├── deeplabv3p.py           
│       ├── fpn.py                  
│       ├── maskrcnn.py            
│       ├── resnet18cnn.py
│       └── unetpp.py               
├── Data/                           # Dataset 
│   ├── Label_2/                    
│   ├── Label_Test_2023April7/     
│   ├── US_2/                       
│   └── US_Test_2023April7/        
└── dump/                           # Additional/older Jupyter notebooks
    ├── bubblesegment.ipynb
    └── Only_testing.ipynb
```

## 5. Experimental Setup

### Dataset
*   B-mode ultrasound images were collected during histotripsy experiments in collaboration with UChicago Radiology.
*   Experiments: Agarose phantoms and ex vivo porcine kidneys.
*   Ultrasound parameters: 1 MHz transducer, 35 MPa, 20 µs pulses @ 50 Hz.
*   Ground truth: Annotated using co-registered digital images.
*   Dataset split: 6 datasets → 880 train / 400 test / holdout set.
    *   The `Data/` directory shows an example structure:
        *   Training Images: `Data/US_2/`
        *   Training Labels: `Data/Label_2/`
        *   Test Images: `Data/US_Test_2023April7/`
        *   Test Labels: `Data/Label_Test_2023April7/`
*   Unlabelled data for unsupervised exploration was provided by Vishwas Trivedi (PhD student, IIT Gandhinagar).
*   **Data Augmentation & Oversampling:** Techniques were applied to increase model robustness, especially for early-pulse frames. Logic for this is within `code_files/dataloader.py`.

## 6. Results
Our proposed method (DeepLabV3 with Dice-Focal Loss) demonstrates significant improvements.

*   **Quantitative Metrics:**
    | Metric          | Our Results | Prior SOTA Results |
    |-----------------|-------------|--------------------|
    | Global Accuracy | 0.97525     | 0.95555            |
    | Mean Accuracy   | 0.90524     | 0.69524            |
    | Mean IoU        | 0.82537     | 0.76461            |
    | Weighted IoU    | 0.93525     | 0.91648            |
    
    *(Evaluation metrics are computed using `code_files/metric.py`)*

*   **Qualitative Results:** 
    ![results](https://github.com/user-attachments/assets/acace566-d5d3-49a5-8ba0-f7e5325b23ea)
    ![visualizations of predictions](https://github.com/user-attachments/assets/367be1c2-6d29-4834-b5ba-bc5e5a1a53ba)


## 7. Challenges Faced
*   Ground truth annotation variability.
*   Difficulty in segmenting sparse bubble regions from initial pulses.
*   Scarcity of annotated histotripsy data.

## 8. Future Work
*   Improve early-pulse segmentation (e.g., using curriculum learning).
*   Integrate attention mechanisms and adaptive loss strategies.
*   Explore unsupervised/semi-supervised learning with limited labels.


## 9. Citation
* Miao K, Basterrechea KF, Hernandez SL, Ahmed OS, Patel MV, Bader KB. Development of Convolutional Neural Network to Segment Ultrasound Images of Histotripsy Ablation. IEEE Trans Biomed Eng. 2024 Jun;71(6):1789-1797. doi: 10.1109/TBME.2024.3352538. Epub 2024 May 20. PMID: 38198256.
* Chen, L., Papandreou, G., Schroff, F., & Adam, H. (2017). Rethinking Atrous Convolution for Semantic Image Segmentation. ArXiv. https://arxiv.org/abs/1706.05587



## 10. Acknowledgments
*   This work was conducted at the [MUSE LAB]([https://sites.google.com/iitgn.ac.in/muse-lab/home](https://labs.iitgn.ac.in/muselaboratory/)), IIT Gandhinagar.
*   We thank UChicago Radiology for collaboration and experimental data.
*   We acknowledge Vishwas Trivedi (IIT Gandhinagar) for unlabelled ultrasound data.

## 11. License
This project is licensed under the MIT License. See the `LICENSE.md` file for details.
