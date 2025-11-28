# Pancreatic Cancer Detection: Hardware-Accelerated Deep Learning

Comparative Analysis of Hardware-Accelerated Deep Learning Models for Early Pancreatic Cancer Detection in Medical Imaging

## 🎯 Overview

This project implements hardware-accelerated deep learning models for early pancreatic cancer detection using endoscopic ultrasound (EUS) imaging. The system compares GPU and FPGA inference performance for CNN-based classification of pancreatic ductal adenocarcinoma (PDAC), healthy pancreas, and pancreatitis conditions. The architecture leverages Google Drive for dataset storage, Python/MATLAB in Google Colab for model development, and HDL code generation targeting Intel FPGA DevCloud for deployment. This comparative analysis aims to demonstrate that FPGA implementations can achieve 2-5× faster inference with lower latency compared to GPU systems while maintaining diagnostic accuracy above 85%.

**Core Hypothesis**: CNN models deployed on FPGA hardware can achieve 2-5× faster inference with lower latency compared to GPU systems while maintaining high diagnostic accuracy (85-95%) for early pancreatic cancer detection.

---

## 🏗️ System Design

### Google Drive
Google Drive serves as the primary storage layer for the 66,249 EUS frames organized into three class-specific folders (CANCER, NORMAL, PANCREATITIS), each containing patient-level subfolders. A master CSV file (`CLINICAL INFORMATION.csv`) indexes all frames with metadata including image path, patient ID, diagnostic label, and train/val/test split assignments. This architecture separates data storage from compute resources, enabling the large dataset (approximately 16.6GB of images) to persist across ephemeral Colab sessions while keeping GitHub repository size manageable.

### Python (Google Colab)
Python scripts running in Colab notebooks handle the data pipeline: mounting Google Drive, reading the master CSV, performing exploratory data analysis, and creating patient-level train/val/test splits. The `pandas` library generates three simplified split CSVs (`train_split_simple.csv`, `val_split_simple.csv`, `test_split_simple.csv`) containing only image paths and labels. These CSVs act as lightweight indices that MATLAB can consume, eliminating redundant data duplication. Python also manages version control by committing split metadata and code to the GitHub repository.

### MATLAB (Colab-based)
MATLAB R2022b runs within the same Colab VM using QEMU x86 emulation on ARM64 hardware. It accesses Drive-stored images via shared `/content/drive/` paths and constructs `imageDatastore` objects from the three split CSVs for training, validation, and testing. The Deep Learning Toolbox trains CNN architectures (ResNet-18 baseline), while HDL Coder and Deep Learning HDL Toolbox convert trained models to synthesizable Verilog/VHDL. Fixed-Point Designer handles int8/int16 quantization for FPGA deployment. Trained models (.mat files) and generated HDL code are saved back to Drive-synced repository folders for reproducibility.

### GitHub
The repository tracks all Python notebooks, MATLAB scripts, documentation, and configuration files. Importantly, the dataset itself remains in Drive—only the split CSV indices are committed to Git, providing reproducibility without bloating the repo. This separation enables rapid iteration: researchers can update split strategies or preprocessing steps by modifying and committing the 200KB CSV files rather than re-uploading gigabytes of images.

### Data Flow Summary

1. **Data Loading**: EUS frames are stored in Drive with per-patient organization; master CSV catalogs all 55 patients across three diagnostic categories.
2. **Indexing**: Python generates train/val/test split CSVs (70/15/15% patient-level split) and commits them to GitHub.
3. **Training**: MATLAB reads split CSVs from the repo, builds datastores pointing to Drive images, and trains CNNs using GPU acceleration (NVIDIA T4).
4. **Quantization**: Fixed-Point Designer converts floating-point models to int8 representation suitable for FPGA resource constraints.
5. **HDL Generation**: Deep Learning HDL Toolbox synthesizes Verilog targeting Intel Arria 10/Stratix 10 architectures available on DevCloud.
6. **Deployment**: Generated HDL is uploaded to Intel FPGA DevCloud for synthesis, place-and-route, and bitstream generation.
7. **Evaluation**: Inference latency and resource utilization are compared between GPU (baseline) and FPGA implementations.

![System Architecture](docs/systemdesign.png)  
*Figure 1: Data flow from Google Drive through Python preprocessing, MATLAB training, and HDL code generation targeting FPGA deployment.*

---

## 🔧 MATLAB in Colab Setup

### Prerequisites
- **MATLAB License**: Valid Individual or Academic license for R2022b covering required toolboxes (Deep Learning, HDL Coder, Deep Learning HDL, Fixed-Point Designer, Computer Vision).
- **License File**: Retrieved `license.lic` file from the MathWorks License Center, stored in Drive or uploaded to Colab session.
- **Colab Environment**: Standard Colab notebook with T4 GPU runtime (free tier sufficient for development; production training may benefit from GCP credits).

### Installation Steps

1. **Install System Dependencies**  
   Run shell commands to install Java (required for MATLAB), QEMU emulation for ARM64→x86 translation, and basic utilities:
   ```bash
   apt-get update -qq
   apt-get install -qq wget unzip openjdk-11-jdk-headless bc qemu-user-static binfmt-support
   ```

2. **Download MATLAB Installer**  
   Fetch the MATLAB Package Manager (MPM) and install R2022b core plus required toolboxes (13GB total):
   ```bash
   wget https://www.mathworks.com/mpm/glnxa64/mpm
   chmod +x mpm
   ./mpm install --release=R2022b --destination=/opt/matlab \
     --products MATLAB Deep_Learning_Toolbox HDL_Coder Deep_Learning_HDL_Toolbox \
     Fixed-Point_Designer Computer_Vision_Toolbox Parallel_Computing_Toolbox
   ```

3. **Configure License**  
   Upload `license.lic` via Colab's file picker, then copy to MATLAB's license directory and set environment variables:
   ```bash
   mkdir -p /opt/matlab/licenses
   cp /content/license.lic /opt/matlab/licenses/
   export MLM_LICENSE_FILE=/opt/matlab/licenses/license.lic
   export LD_LIBRARY_PATH=/opt/matlab/bin/glnxa64:$LD_LIBRARY_PATH
   ```

4. **Verify Installation**  
   Test MATLAB launch in batch mode to confirm toolboxes are accessible:
   ```bash
   matlab -batch "ver; gpuDevice; disp('MATLAB ready'); exit"
   ```

5. **Persist Environment Variables**  
   Add exports to `.bashrc` to streamline setup scripts:
   ```bash
   echo 'export MLM_LICENSE_FILE=/opt/matlab/licenses/license.lic' >> ~/.bashrc
   echo 'export LD_LIBRARY_PATH=/opt/matlab/bin/glnxa64:$LD_LIBRARY_PATH' >> ~/.bashrc
   ```

### Important Notes
- **Ephemeral Storage**: Colab VMs reset every 12 hours or on disconnect. The above steps must re-run at the start of each session. Store the notebook itself in Drive to preserve installation commands.
- **Licensing**: This guide does not include license keys or proprietary installer commands. Refer to the [notebook cells](https://github.com/adlikestocode/pancreatic-cancer-fpga/blob/main/pancreaticdetection.ipynb) for exact implementation details.
- **Performance**: MATLAB's x86 binaries run via QEMU emulation, introducing ~10-20% overhead compared to native execution. For production workloads, consider using x86 Colab instances or local MATLAB installations.

---

## 📊 Dataset

**Endoscopic Ultrasound Database of the Pancreas**
- **Source**: [Ecoendoscopy Database - Google Drive](https://drive.google.com/drive/folders/10GPl3r-ppDyWwWzneoSFH52yxUGX4xkw)
- **Total Patients**: 55 (18 cancer, 32 healthy, 5 pancreatitis)
- **Total EUS Frames**: 66,249 annotated images
- **Cancer Cases**: C01-C18 with confirmed histological diagnosis
- **Controls**: H01-H32 (healthy pancreas)
- **Clinical Metadata**: TNM staging, tumor size, lesion localization, pathology reports

### Dataset Statistics
- Cancer frames: 16,585 (25.0% - PDAC confirmed via biopsy)
- Healthy frames: 42,190 (63.7% - normal pancreas)
- Pancreatitis frames: 7,474 (11.3% - chronic pancreatitis)
- Age range: 26-87 years
- Tumor sizes: 15-43.8mm (mean ~31mm)

### Data Split (Patient-Level)
- **Training**: 45,556 frames (68.8%) - 12 cancer, 22 healthy, 3 pancreatitis patients
- **Validation**: 13,015 frames (19.6%) - 3 cancer, 5 healthy, 1 pancreatitis patient
- **Testing**: 7,678 frames (11.6%) - 3 cancer, 5 healthy, 1 pancreatitis patient

*Note: Split performed at patient level to prevent data leakage between sets.*

---

## 🛠️ Tech Stack

### Development Environment
- **Version Control**: GitHub
- **Development Platform**: Google Colab (T4 GPU, free tier)
- **Cloud Compute**: Google Cloud Platform (GCP with $300 education credits)
- **FPGA Cloud**: Intel FPGA DevCloud (free for academics)

### Software & Frameworks
- **Primary**: MATLAB R2022b (Academic License)
  - Deep Learning Toolbox
  - HDL Coder Toolbox
  - Deep Learning HDL Toolbox
  - Fixed-Point Designer
  - Computer Vision Toolbox
  - Parallel Computing Toolbox
- **Data Processing**: Python 3.8+ (Pandas, NumPy, Matplotlib)
- **HDL Synthesis**: Vivado/Quartus (Intel DevCloud pre-installed)

### Hardware Targets
- **GPU Baseline**: NVIDIA T4 (Google Colab/GCP)
- **FPGA Deployment**: Intel Arria 10 / Stratix 10 (DevCloud)
- **Alternative**: Xilinx Virtex UltraScale+ (AWS F1 if needed)

---

## 📅 Development Timeline

### Sprint 1: Data Pipeline ✅ COMPLETED
**Goal**: Organize and preprocess EUS dataset for training

- [x] **Task 1.1**: Clone dataset from Google Drive to GitHub repo
- [x] **Task 1.2**: Load clinical metadata (CSV parsing)
- [x] **Task 1.3**: Exploratory data analysis (tumor distribution, class balance)
- [x] **Task 1.4**: Organize images into train/val/test splits (70/15/15)
- [x] **Task 1.5**: Data preprocessing (resize to 224×224, normalization)
- [x] **Deliverable**: `data/train_val_test_split.csv` with patient-level splits

**Tools**: Python (Pandas), Google Colab

---

### Sprint 2: CNN Training & Validation (Nov 28-29)
**Goal**: Train and validate CNN model for 3-class classification

- [ ] **Task 2.1**: Setup MATLAB in Google Colab with academic license (30 min)
- [ ] **Task 2.2**: Configure ResNet-18 transfer learning architecture (30 min)
- [ ] **Task 2.3**: Train CNN on T4 GPU (10 epochs, data augmentation) (2 hours)
- [ ] **Task 2.4**: Validation metrics (accuracy, precision, recall, F1, confusion matrix) (1 hour)
- [ ] **Deliverable**: Trained model (`models/trained_cnn.mat`), validation report

**Target**: 85%+ accuracy on test set (matching pilot study benchmarks)

**Tools**: MATLAB Deep Learning Toolbox, Colab T4 GPU

---

### Sprint 3: GPU Benchmarking (Nov 29)
**Goal**: Establish GPU inference baseline

- [ ] **Task 3.1**: Apply for GCP education credits ($300) (10 min)
- [ ] **Task 3.2**: Deploy trained model to GCP T4/V100 instance (30 min)
- [ ] **Task 3.3**: Run inference latency tests (1000+ images) (1 hour)
- [ ] **Task 3.4**: Profile throughput, memory usage, power consumption (1 hour)
- [ ] **Deliverable**: GPU benchmark results (`benchmarks/gpu/metrics.csv`)

**Target**: 10-20ms inference latency per image

**Tools**: GCP Compute Engine, MATLAB

---

### Sprint 4: FPGA HDL Conversion (Nov 30)
**Goal**: Generate and synthesize HDL from trained CNN

- [ ] **Task 4.1**: Apply for Intel FPGA DevCloud access (10 min)
- [ ] **Task 4.2**: Quantize CNN model to fixed-point (int8/int16) (1 hour)
- [ ] **Task 4.3**: Generate HDL code using `dlhdl.Workflow` (1 hour)
- [ ] **Task 4.4**: Verify generated HDL testbenches (30 min)
- [ ] **Deliverable**: HDL code (`hdl_output/`), quantization report

**Tools**: MATLAB HDL Coder, Fixed-Point Designer

---

### Sprint 5: FPGA Deployment & Benchmarking (Dec 1)
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

### Sprint 6: Results & Documentation (Dec 2)
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
│   └── train_val_test_split.csv    # Patient-level data split index
├── scripts/
│   ├── 01_data_organization.py     # Sprint 1: Data loading & splitting
│   ├── 02_preprocessing.py         # Sprint 1: Image preprocessing
│   ├── 03_training.m               # Sprint 2: CNN training
│   ├── 04_validation.m             # Sprint 2: Metrics calculation
│   ├── 05_gpu_benchmark.m          # Sprint 3: GPU inference profiling
│   ├── 06_hdl_conversion.m         # Sprint 4: HDL generation
│   └── 07_fpga_deploy.m            # Sprint 5: FPGA synthesis & testing
├── models/
│   ├── trained_cnn.mat             # Trained ResNet-18 model
│   └── quantized_cnn.mat           # Fixed-point quantized model
├── hdl_output/                     # Generated HDL code
├── benchmarks/
│   ├── gpu/                        # GPU latency results
│   ├── fpga/                       # FPGA synthesis reports
│   └── comparison/                 # GPU vs FPGA plots
├── docs/
│   ├── systemdesign.jpg            # System architecture diagram
│   ├── project_report.pdf          # Final research paper
│   └── references.bib              # Bibliography
├── pancreaticdetection.ipynb       # Main Colab notebook
└── README.md                       # This file
```

---

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

---

## 📚 References

1. Cui et al. (2021) - Early screening and diagnosis strategies of pancreatic cancer
2. Ozkan et al. (2016) - Deep learning analysis for PDAC detection on endosonographic images
3. Tonozuka et al. (2021) - Real-time CAD for focal pancreatic masses using CNN-LSTM
4. Gajos & Chetty (2020) - Endoscopic ultrasound database of the pancreas
5. Gambino et al. (2020) - Fixed-point code synthesis for neural networks

---

## 🤝 Acknowledgments

- **Dataset**: Endoscopic Ultrasound Database of the Pancreas
- **Cloud Resources**: Google Cloud Platform Education Credits, Intel FPGA DevCloud
- **Tools**: MATLAB Academic License, GitHub Student Developer Pack

---

## 📧 Contact

**Repository**: [github.com/adlikestocode/pancreatic-cancer-fpga](https://github.com/adlikestocode/pancreatic-cancer-fpga)  
**Author**: Aditya (adlikestocode)  
**Email**: adityavkini3004@gmail.com

---

**Status**: 🚧 Active Development (Sprint 1 Complete, Sprint 2 In Progress)  
**Last Updated**: November 28, 2025
