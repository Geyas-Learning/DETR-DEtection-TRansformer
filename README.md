# 🤖 detection_detr

## Project Overview

This repository contains the implementation of [briefly state what your project does, e.g., an object detection model] based on the **DETR (DEtection TRansformer)** architecture.

The project focuses on [mention the specific task or dataset, e.g., detecting objects in a custom aerial imagery dataset or adapting the model for a specific domain].

## ⚙️ Installation and Setup

### Prerequisites

* Python 3.9
conda create -p ./detr_env python=3.9 -y

### Environment Setup

1.  **Clone the repository:**
    ```bash
    mkdir cvia
    cd cvia
    mkdir project_detr
    cd project_detr
    git clone [https://github.com/Geyas-Learning/detection_detr.git](https://github.com/Geyas-Learning/detection_detr.git)
    
    ```

2.  **Create a virtual environment** (recommended):
    ```bash
    create env in the cvia folder
    conda create -p ./detr_env python=3.9 -y
    conda activate /mnt/aiongpfs/users/gbanisetty/cvia/detr_env
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt 
    # If you don't have a requirements.txt, list key packages here:
    # pip install torch torchvision transformers
    ```
```bash
    Package                  Version
------------------------ -----------
absl-py                  2.3.1
contourpy                1.3.0
cycler                   0.12.1
filelock                 3.19.1
fonttools                4.60.1
fsspec                   2025.10.0
grpcio                   1.76.0
importlib_metadata       8.7.0
importlib_resources      6.5.2
Jinja2                   3.1.6
joblib                   1.5.2
kiwisolver               1.4.7
Markdown                 3.9
MarkupSafe               3.0.3
matplotlib               3.9.4
mpmath                   1.3.0
networkx                 3.2.1
numpy                    2.0.2
nvidia-cublas-cu12       12.8.4.1
nvidia-cuda-cupti-cu12   12.8.90
nvidia-cuda-nvrtc-cu12   12.8.93
nvidia-cuda-runtime-cu12 12.8.90
nvidia-cudnn-cu12        9.10.2.21
nvidia-cufft-cu12        11.3.3.83
nvidia-cufile-cu12       1.13.1.3
nvidia-curand-cu12       10.3.9.90
nvidia-cusolver-cu12     11.7.3.90
nvidia-cusparse-cu12     12.5.8.93
nvidia-cusparselt-cu12   0.7.1
nvidia-nccl-cu12         2.27.3
nvidia-nvjitlink-cu12    12.8.93
nvidia-nvtx-cu12         12.8.90
opencv-python            4.12.0.88
packaging                25.0
pandas                   2.3.3
pillow                   11.3.0
pip                      25.3
protobuf                 6.33.1
pycocotools              2.0.10
pyparsing                3.2.5
python-dateutil          2.9.0.post0
pytz                     2025.2
scikit-learn             1.6.1
scipy                    1.13.1
setuptools               80.9.0
six                      1.17.0
sympy                    1.14.0
tensorboard              2.20.0
tensorboard-data-server  0.7.2
tensorboardX             2.6.4
threadpoolctl            3.6.0
torch                    2.8.0
torchvision              0.23.0
triton                   3.4.0
typing_extensions        4.15.0
tzdata                   2025.2
Werkzeug                 3.1.3
wheel                    0.45.1
zipp                     3.23.0

```

## 🏃 Usage

### Training

To train the DETR model on your dataset, run the following command:

```bash
cd project_detr
sbatch detection_detr/run_DR50.sh
sbatch detection_detr/run_DR152.sh
```

#### Tree structure for the files and folder

```bash
cvia/
├── detr_env/                # (conda environment - create here)
├── logs/
│
├── project_detr/
│   ├── __pycache__/
│   ├── data/
│   │   ├── images/
│   │   ├── mask/
│   │   ├── train.csv
│   │   ├── val.csv
│   │  
│   │
│   ├── detection_detr/
│   │   ├── __pycache__/
│   │   ├── detr_model/............
│   │   │
│   │   ├── util/.........
│   │   │
│   │   ├── coco_custom.py
│   │   ├── data_utils.py
│   │   ├── DR50_main.py
│   │   ├── DR50_pipeline.py
│   │   ├── DR152_main.py
│   │   ├── DR152_pipeline.py
|   │   ├── README.md
|   │   ├── requirements.txt
|   │   ├── run_DR50.sh
|   │   └── run_DR152.sh
│   |
├   |── DR50/        #output will generate in this folder after execution
│   |
│   |
├   |── DR150/       #output will generate in this folder after execution    
│   |     
│   | 
|   └── test/        #detection test dataset folder
|         ├── images/   
|

```

