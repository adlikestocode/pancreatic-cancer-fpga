# Pancreatic Cancer Detection: Hardware-Accelerated Deep Learning

Comparative Analysis of Hardware-Accelerated Deep Learning Models for Early Pancreatic Cancer Detection in Medical Imaging

## 🎯 Project Overview

This project implements and benchmarks CNN-based pancreatic ductal adenocarcinoma (PDAC) detection using endoscopic ultrasound (EUS) imaging, comparing GPU and FPGA hardware acceleration for clinical deployment.

**Core Hypothesis**: CNN models deployed on FPGA hardware can achieve 2-5× faster inference with lower latency compared to GPU systems while maintaining high diagnostic accuracy (85-95%) for early pancreatic cancer detection.

## 📊 Dataset

**Endoscopic Ultrasound Database of the Pancreas**
- **Source**: [Ecoendoscopy Database - Google Drive](https://drive.google.com/drive/folders/10GPl3r-ppDyWwWzneoSFH52yxUGX4xkw)
- **Total Patients**: 55 (18 cancer, 32 healthy, 5 pancreatitis)
- **Total EUS Frames**: ~53,000+ annotated images
- **Cancer Cases**: C01-C18 with confirmed histological diagnosis
- **Controls**: H01-H32 (healthy pancreas)
- **Clinical Metadata**: TNM staging, tumor size, lesion localization, pathology reports

### Dataset Statistics
- Cancer frames: ~13,000+ (PDAC confirmed via biopsy)
- Healthy frames: ~40,000+ (normal pancreas)
- Pancreatitis frames: ~7,000+ (chronic pancreatitis)
- Age range: 26-87 years
- Tumor sizes: 15-43.8mm (mean ~31mm)

## 🛠️ Tech Stack

### Development Environment
- **Version Control**: GitHub
- **Development Platform**: Google Colab (T4 GPU, free tier)
- **Cloud Compute**: Google Cloud Platform (GCP with $300 education credits)
- **FPGA Cloud**: Intel FPGA DevCloud (free for academics)

### Software & Frameworks
- **Primary**: MATLAB R2023b+ (Academic License)
  - Deep Learning Toolbox
  - HDL Coder Toolbox
  - Fixed-Point Designer
- **Data Processing**: Python 3.8+ (Pandas, NumPy, Matplotlib)
- **HDL Synthesis**: Vivado/Quartus (Intel DevCloud pre-installed)

### Hardware Targets
- **GPU Baseline**: NVIDIA T4 (Google Colab/GCP)
- **FPGA Deployment**: Intel Arria 10 / Stratix 10 (DevCloud)
- **Alternative**: Xilinx Virtex UltraScale+ (AWS F1 if needed)

## 📅 Development Timeline

### Sprint 1: Data Pipeline (Nov 27, Evening - 4 hours)
**Goal**: Organize and preprocess EUS dataset for training

- [ ] **Task 1.1**: Clone dataset from Google Drive to GitHub repo (30 min)
- [ ] **Task 1.2**: Load clinical metadata (CSV parsing) (20 min)
- [ ] **Task 1.3**: Exploratory data analysis (tumor distribution, class balance) (30 min)
- [ ] **Task 1.4**: Organize images into train/val/test splits (70/15/15) (1 hour)
- [ ] **Task 1.5**: Data preprocessing (resize to 224×224, normalization) (45 min)
- [ ] **Deliverable**: `data/` directory with organized imageDatastore structure

**Tools**: Python (Pandas), Google Colab

---

### Sprint 2: CNN Training & Validation (Nov 28, Morning - 4 hours)
**Goal**: Train and validate CNN model for binary classification

- [ ] **Task 2.1**: Setup MATLAB in Google Colab with academic license (30 min)
- [ ] **Task 2.2**: Configure ResNet-18 transfer learning architecture (30 min)
- [ ] **Task 2.3**: Train CNN on T4 GPU (10 epochs, data augmentation) (2 hours)
- [ ] **Task 2.4**: Validation metrics (accuracy, precision, recall, F1, confusion matrix) (1 hour)
- [ ] **Deliverable**: Trained model (`models/trained_cnn.mat`), validation report

**Target**: 85%+ accuracy on test set (matching pilot study benchmarks)

**Tools**: MATLAB Deep Learning Toolbox, Colab T4 GPU

---

### Sprint 3: GPU Benchmarking (Nov 28, Afternoon - 3 hours)
**Goal**: Establish GPU inference baseline

- [ ] **Task 3.1**: Apply for GCP education credits ($300) (10 min)
- [ ] **Task 3.2**: Deploy trained model to GCP T4/V100 instance (30 min)
- [ ] **Task 3.3**: Run inference latency tests (1000+ images) (1 hour)
- [ ] **Task 3.4**: Profile throughput, memory usage, power consumption (1 hour)
- [ ] **Deliverable**: GPU benchmark results (`benchmarks/gpu/metrics.csv`)

**Target**: 10-20ms inference latency per image

**Tools**: GCP Compute Engine, MATLAB

---

### Sprint 4: FPGA HDL Conversion (Nov 29, Morning - 4 hours)
**Goal**: Generate and synthesize HDL from trained CNN

- [ ] **Task 4.1**: Apply for Intel FPGA DevCloud access (10 min)
- [ ] **Task 4.2**: Quantize CNN model to fixed-point (int8/int16) (1 hour)
- [ ] **Task 4.3**: Generate HDL code using `dlhdl.Workflow` (1 hour)
- [ ] **Task 4.4**: Verify generated HDL testbenches (30 min)
- [ ] **Deliverable**: HDL code (`hdl_output/`), quantization report

**Tools**: MATLAB HDL Coder, Fixed-Point Designer

---

### Sprint 5: FPGA Deployment & Benchmarking (Nov 29, Afternoon - 5 hours)
**Goal**: Deploy to FPGA and compare against GPU

- [ ] **Task 5.1**: Upload HDL to Intel DevCloud (30 min)
- [ ] **Task 5.2**: Synthesize on Arria 10 / Stratix 10 (2 hours)
- [ ] **Task 5.3**: Run FPGA inference tests (same 1000 image set) (1 hour)
- [ ] **Task 5.4**: Analyze resource utilization (LUTs, DSPs, BRAM) (30 min)
- [ ] **Task 5.5**: Generate GPU vs FPGA comparison (latency, throughput, power) (1 hour)
- [ ] **Deliverable**: FPGA benchmark results, comparison plots

**Target**: 5-10ms latency, 2-5× speedup vs GPU

**Tools**: Intel Quartus, FPGA DevCloud, MATLAB

---

### Sprint 6: Results & Documentation (Nov 30 - 3 hours)
**Goal**: Complete paper sections 4-5, finalize repository

- [ ] **Task 6.1**: Write experimental results section with metrics (1 hour)
- [ ] **Task 6.2**: Create comparison visualizations (latency plots, resource charts) (1 hour)
- [ ] **Task 6.3**: Update README with final results (30 min)
- [ ] **Task 6.4**: Code cleanup, add inline documentation (30 min)
- [ ] **Deliverable**: Complete research paper draft, polished GitHub repo

**Tools**: MATLAB (plotting), Markdown

---

## 📁 Repository Structure

```
pancreatic-cancer-fpga/
├── data/
│   ├── raw/                    # Original EUS images (C01-C18, H01-H32, P01-P05)
│   ├── processed/              # Resized & normalized images
│   ├── train/                  # Training set (70%)
│   ├── val/                    # Validation set (15%)
│   └── test/                   # Test set (15%)
├── scripts/
│   ├── 01_data_organization.py # Sprint 1: Data loading & splitting
│   ├── 02_preprocessing.py     # Sprint 1: Image preprocessing
│   ├── 03_training.m           # Sprint 2: CNN training
│   ├── 04_validation.m         # Sprint 2: Metrics calculation
│   ├── 05_gpu_benchmark.m      # Sprint 3: GPU inference profiling
│   ├── 06_hdl_conversion.m     # Sprint 4: HDL generation
│   └── 07_fpga_deploy.m        # Sprint 5: FPGA synthesis & testing
├── models/
│   ├── trained_cnn.mat         # Trained ResNet-18 model
│   └── quantized_cnn.mat       # Fixed-point quantized model
├── hdl_output/                 # Generated HDL code
├── benchmarks/
│   ├── gpu/                    # GPU latency results
│   ├── fpga/                   # FPGA synthesis reports
│   └── comparison/             # GPU vs FPGA plots
├── docs/
│   ├── project_report.pdf      # Final research paper
│   └── references.bib          # Bibliography
├── CLINICAL-INFORMATION.csv    # Patient metadata
└── README.md                   # This file
```

## 🎯 Success Metrics

### Model Performance
- **Accuracy**: ≥85% on test set (target: match pilot studies at 85-95%)
- **Sensitivity**: ≥90% (minimize false negatives for cancer detection)
- **Specificity**: ≥80% (reduce false positives)

### Hardware Benchmarks
- **GPU Latency**: 10-20ms per image (T4 baseline)
- **FPGA Latency**: 5-10ms per image (target: 2-5× speedup)
- **Resource Utilization**: <85% LUTs, <70% DSPs on Arria 10
- **Power Efficiency**: Document FPGA power vs GPU (if available)

## 📚 References

1. Cui et al. (2021) - Early screening and diagnosis strategies of pancreatic cancer
2. Ozkan et al. (2016) - Deep learning analysis for PDAC detection on endosonographic images
3. Tonozuka et al. (2021) - Real-time CAD for focal pancreatic masses using CNN-LSTM
4. Gajos & Chetty (2020) - Endoscopic ultrasound database of the pancreas
5. Gambino et al. (2020) - Fixed-point code synthesis for neural networks

## 🤝 Acknowledgments

- Dataset: Endoscopic Ultrasound Database of the Pancreas
- Cloud Resources: Google Cloud Platform Education Credits, Intel FPGA DevCloud
- Tools: MATLAB Academic License, GitHub Student Developer Pack

## 📧 Contact

**Repository**: [github.com/adlikestocode/pancreatic-cancer-fpga](https://github.com/adlikestocode/pancreatic-cancer-fpga)  
**Author**: Aditya (adlikestocode)  
**Email**: adityavkini3004@gmail.com

---

**Status**: 🚧 Active Development (Sprint 1 in progress)  
**Last Updated**: November 27, 2025