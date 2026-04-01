This repository is an **unofficial implementation** of the paper:

> "Service recommendation based on contrastive learning and multi-task learning”\
> *Published in 2024* *Computer Communications*

The code was implemented by **Xinrou Kang**👩‍💻, based on the concepts and methods described in the original paper. It is not an official release by the original authors.

## Installation Guide

1.  Create and activate virtual environment (recommended)

```bash
conda create -n api_rec python=3.8
conda activate api_rec
```

1.  Install dependencies

```bash
pip install -r requirements.txt
```

## Quick Start

1.  Data Preprocessing

```bash
python data/preprocessing/Predata_DNN.py
```

1.  Model Training and Evaluation

```bash
# Run 10-fold cross-validation
python main.py

# Merge recommendation results
python merge.py
```

1.  View Results

```bash
# Final recommendations will be saved in
output/merged_recommend_data.json
```

