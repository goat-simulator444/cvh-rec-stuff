This repository is an **unofficial implementation** of the ICWS paper:

> **"Biased Random Walk based Web API Recommendation in Heterogeneous Network"**\
> Published in 2024 IEEE International Conference on Web Services (ICWS)

The code was implemented by **Yihui Wang**👩‍💻, based on the concepts and methods described in the original paper. It is not an official release by the original authors.

## 🗂 Project Structure

```

├── data/
│   ├── api_id_mapping.json            # API information file
│   └── shuffle_mashup_details.json    # Mashup-API usage records (shuffled)
├── data_loader.py
├── graph_model.py
├── random_walk_node2vec.py
├── train_eval.py
├── requirements.txt
└── README.md

```

## 🚀 Getting Started

### 1. Install dependencies

    pip install -r requirements.txt

### 2. Run the main script

    python train_eval.py

The script will load data, build the graph, perform biased random walks, train embeddings, and generate the recommendation results.

## 📁 Data

Place the following files in the `data/` directory:

*   `api_id_mapping.json`: Mapping between API names and IDs

*   `shuffle_mashup_details.json`: Records of Mashup and the APIs they use (shuffled)

