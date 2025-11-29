# Colab → GitHub Workflow

This project develops in Google Colab and versions artifacts in this GitHub repository. Colab helper cells push CSVs (data splits), notebooks, and MATLAB .m files securely using Git.

## Colab Repo Layout
```
/content/pancreatic-cancer-fpga
├── data/       # CSV split indices (no raw images)
├── notebooks/  # Colab notebooks
└── matlab/     # MATLAB DAG pipeline (.m files)
```

## Authentication via Colab Secrets
Avoid hard-coding tokens by using Colab's Secrets feature:

1. Open Colab left sidebar → "Secrets"
2. Add: Name `GITHUB_TOKEN`, Value (GitHub PAT with repo scope), Keep "Hide secret" enabled

Notebook code fetches and uses it safely:
```python
import os, subprocess
from google.colab import userdata

os.chdir("/content/pancreatic-cancer-fpga")
token = userdata.get('GITHUB_TOKEN')
if not token:
    raise ValueError("GITHUB_TOKEN not found in Colab secrets.")

remote_url = f"https://{token}@github.com/adlikestocode/pancreatic-cancer-fpga.git"
subprocess.run(["git", "remote", "set-url", "origin", remote_url], check=True, text=True)
```
Token stays hidden from code and Git history.

## Staging and Pushing
After generating splits and .m files:
```python
import os
os.chdir("/content/pancreatic-cancer-fpga")

# Stage
!git add data/train_split_simple.csv data/val_split_simple.csv data/test_split_simple.csv
!git add notebooks/pancreaticdetection.ipynb
!git add matlab/*.m 2>/dev/null || echo "No MATLAB .m files found"

# Commit
commit_message = "Update MATLAB DAG pipeline, splits, and notebook"
os.system(f'git commit -m "{commit_message}"')

# Push
os.system("git push origin main")
```

On GCP/VM: `git clone https://github.com/adlikestocode/pancreatic-cancer-fpga.git` (no token needed).

## MATLAB DAG Pipeline
Core logic splits into modular .m files under `matlab/` (DAG-style, not one big script):

- `01_datageneration.m`: Patient-level train/val/test CSV splits
- `02_preprocessing.m`: Grayscale→RGB, resize, normalize, augment
- `03_training.m`: Train ResNet-18 CNN
- `04_validation.m`: Metrics, confusion matrices
- `05_deploy_gpu.m`: GPU inference prep/benchmark
- `06_deploy_fpga.m`: HDL Coder/FPGA prep
- `07_analysis.m`: Results summary

## Why DAG Design?
- **Transparency**: Single-responsibility files pinpoint issues fast
- **Error handling/restart**: Fail at training? Rerun from `03_training.m` only
- **Modularity**: Swap models or preprocessing without touching upstream/downstream

## Wrapper Execution Example
Run sequentially:
```matlab
addpath('matlab');
run('01_datageneration.m');
run('02_preprocessing.m');
run('03_training.m');
run('04_validation.m');
run('05_deploy_gpu.m');
% run('06_deploy_fpga.m');  % Uncomment for FPGA
run('07_analysis.m');
```
Each stage outputs Git-trackable artifacts for reproducibility.
