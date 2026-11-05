# Phase I: Behavioral and Diagnostic Model Analysis (VIS4AI)

**Course:** AI and Visualization Lab (WS26) – Visual Data Science Lab, Johannes Kepler University Linz (JKU)  
**Repository:** [https\://github.com/AndresFigueroa24/AI\_visualization](https://github.com/AndresFigueroa24/AI_visualization)  
**Authors:** Andrés Figueroa, Simon Mader, Enkhdalai Munkhdorj and Karin Wagner  
**Inspected Architecture:** `YOLO26n` (Black-Box Object Detector)  
**Target Benchmark:** Pascal VOC (`VOC2007` & `VOC2012`)

---

## 📌 1\. Project Overview & Scope

This project implements **Phase I: How does the model behave?** within the VIS4AI framework.

### Black-Box Diagnostic Methodology

In accordance with the assignment guidelines:

- **Zero Internal Access:** The target neural network is treated strictly as an opaque black box ($f(X)\rightarrow \hat{Y}$). No internal layer weights, feature activations, or saliency maps (e.g., Grad-CAM) are inspected in this phase (reserved strictly for Phase II).  
- **Behavioral Input-Output Mapping:** We investigate recurring patterns, error modes, and systematic biases by observing how changes in input $X$ directly translate to changes in predictions $\hat{Y}$, confidence scores $c$, and bounding box localization metrics ($IoU$).  
- **AI4VIS Integration:** Generative AI was leveraged as a collaborative assistant to brainstorm visual encodings, prototype diagnostic metrics, and accelerate development. A summary of prompts and design iterations is recorded in Section 6\.

---

## 📁 2\. Repository Layout

```
AI_visualization/
├── .gitignore
├── team-info.json                 # Mandatory course team registration
├── workload.json                  # Student contribution distribution
├── README.md                      # Primary project documentation
└── phase1/
    ├── offline_test.py            # Batch evaluation, IoU matching & CSV exporter
    ├── visualization.py           # Streamlit interactive diagnostic dashboard
    ├── voc_behavioral_results.csv # Precomputed diagnostic benchmark results
    ├── yolo26n.pt                 # Model checkpoint weights (or yolov8n.pt fallback)
    └── VOC_dataset/
        └── VOCdevkit/
            ├── VOC2007/
            │   ├── Annotations/   # Pascal VOC XML ground-truth labels
            │   └── JPEGImages/    # VOC2007 test and train images (.jpg)
            └── VOC2012/
                ├── Annotations/   # Pascal VOC XML ground-truth labels
                └── JPEGImages/    # VOC2012 test and train images (.jpg)
```

---

## ⚙️ 3\. Environment & Prerequisites

The codebase is tested on **Python 3.10+ / Python 3.13**.

### Quick Install

Install all required libraries via `pip`:

```powershell
pip install streamlit ultralytics plotly pandas numpy pillow
```

&nbsp;

### Model Checkpoint

Ensure your weights file (`yolo26n.pt`) resides inside the `phase1/` folder. If running base weights as a fallback:

```powershell
Invoke-WebRequest -Uri "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt" -OutFile "phase1/yolov8n.pt"
```

&nbsp;

---

## 📋 4\. Metadata Files

### `team-info.json` (Root Directory)

```json
{
    "team_name": "Team Name",
    "students": [
        {
            "student_id": "k12540435",
            "first_name": "Andres",
            "last_name": "Figueroa Yamunaque",
            "email": "afigueroa2409@gmail.com",
            "github_user": "AndresFigueroa24"
        },
		{
            "student_id": "k12105337",
            "first_name": "Simon",
            "last_name": "Mader",
            "email": "1simon.mader1@gmail.com",
            "github_user": "Simon14052002"
        },
		        {
            "student_id": "k12542259",
            "first_name": "Enkhdalai",
            "last_name": "Munkhdorj",
            "email": "enkhdalai.munkh@mail.com",
            "github_user": "enkhdalaimunkh-dotcom"
        },
		        {
            "student_id": "k00000000",
            "first_name": "Karin",
            "last_name": "Wagner",
            "email": "first.last@mail.com",
            "github_user": "mygitusername"
        },
    ]
}
```

### `workload.json` (Root Directory)

```json
{
    "team_name": "Team Name",
    "students": [
        {
            "first_name": "Andres",
            "last_name": "Figueroa Yamunaque",
            "workload": 0.25
        },
		{
            "first_name": "Simon",
            "last_name": "Mader",
            "workload": 0.25
        },
		{
            "first_name": "Enkhdalai",
            "last_name": "Munkhdorj",
            "workload": 0.25
        },
		{
            "first_name": "Karin",
            "last_name": "Wagner",
            "workload": 0.25
        },
    ]
} 
```

&nbsp;