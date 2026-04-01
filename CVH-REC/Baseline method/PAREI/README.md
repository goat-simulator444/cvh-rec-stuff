## 1. PAREI

PAREI is a progressive Web API recommendation method that combines explicit and implicit information. The method integrates:

*   Text Similarity (BM25)

*   Graph Structure Features (Node2Vec)

*   Semantic Similarity (SimCSE)

> Reference: Ye Wang, Aohui Zhou, Qiao Huang, Xiaoyang Wang, Bo Jiang. "PAREI: A progressive approach for Web API recommendation by combining explicit and implicit information." Information and Software Technology, 2023.

## Installation Guide

1.  Create and activate virtual environment (recommended)

```bash
conda create -n PAREI python=3.8
conda activate PAREI
```

1.  Install dependencies

```bash
pip install -r requirements.txt
pip install torch==2.0.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html

```

## Quick Start

### PAREI Model

1.  Data Preprocessing

```bash
# 1. Run basic data preprocessing
python data/preprocessing/Data_preprocessing.py

# 2. Generate Node2Vec features
python data/preprocessing/node2vec_mashup.py

# 3. Process 10-fold cross-validation data
python data/preprocessing/processing_10_fold.py
```

1.  Model Training and Evaluation

```bash
# Run main model
python PAREI.py

# Process results
python process_result.py
```

1.  View Results

```bash
# Final recommendations will be saved in
PAREI/output_recommendations.json
```

