# CVH-REC: A novel method for web API recommendation based on the cross-view HGNNs

This repository contains the dataset and source code for the paper:\
**"CVH-REC: A novel method for web API recommendation based on the cross-view HGNNs"**

## 1. Environment Setup

Our code has been tested under **Python 3.12.3**, and the following packages are required:

    h5py==3.11.0 
    scikit_learn==1.4.2 
    sentence_transformers==2.7.0 
    torch==2.7.0+cu126

We recommend setting up the environment using **Conda** to ensure compatibility and ease of installation.

#### ✅ Step 1: Create and activate a new Conda environment

    conda create -n CVH-Rec python=3.12.3 -y
    conda activate CVH-Rec 

#### ✅ Step 2: Install the required Python packages

Install all dependencies (except PyTorch) via `requirements.txt`:

    pip install -r requirements.txt

> ⚠️ Note: The `requirements.txt` file includes all necessary libraries **except for PyTorch**, which should be installed separately to match your system and CUDA version.

#### ✅ Step 3: Install PyTorch manually

Visit the official PyTorch installation page [here](https://pytorch.org/) to choose the correct version for your environment.\
For example, to install **PyTorch with CUDA 12.6**, use:

    pip install torch --index-url https://download.pytorch.org/whl/cu126

## 2. Usage

### 2.1 Data generation (Optional)

> This step can be skipped — preprocessed data is already provided.\
> Run this only if you want to **apply CVH-REC on a new dataset**.

To generate the dataset, start Python in **command line** mode and run:

    python generate_dataset.py

This script uses the original files `Original Dataset/mashups.json` and `Original Dataset/apis.json` as input and performs the following operations:

*   Generates **training set, testing set, and validation set**  under the `dataset` folder.

*   Creates a `data` folder containing:

    *   `API_vectors.h5` and `vectors.h5`: Semantic embedding vectors for API and mashup descriptions.

    *   `api_tag_vector.h5` and `mashup_tag_vector.h5`: Semantic embedding vectors for API and mashup tags.

Each `dataset/foldX/` directory includes:

    RS.csv        # Test set
    TE.csv        # Training set
    VA.csv        # Validation set
    api_tags.csv  # API–tag relationships
    mashup_tags.csv # Mashup–tag relationships

### 2.2 Reproducing Experiments in the Paper

#### 🔍 RQ1: Performance of CVH-REC

##### ▶️ Step 1. **Train the model and generate recommendation results for a single fold (e.g., fold1)**

    python main.py --dataset fold1 --patience 5 --epoch 200 --lr 0.002 --alpha1 0.6 --alpha2 0.6 --alpha3 0.6 --alpha4 0.6

    python main_text_encoder.py --dataset fold1 --method CVH-REC  --patience 5 --lr 0.002 --epoch 200

After training completes, the **recommendation results** for `fold1` will be automatically saved in **JSON format** under the ../`output/` directory.

##### 🔁 Cross-Validation

To perform **10-fold cross-validation**, repeat **Step 1** for each fold by updating the `--dataset` parameter:

*   `--dataset fold1`

*   `--dataset fold2`

*   …

*   `--dataset fold10`

> All other parameters (e.g., learning rate, patience, alpha values) can remain unchanged.

##### ▶️ Step 2. **Compute evaluation metrics (after all 10 folds are completed)**

Once all recommendation result of **10 folds** generated, run the following command to compute the overall performance:

    python metrics.py --method CVH-REC

> ⚠️ **Important:**\
> Do **not** run this step until all 10 folds have finished training.\
> If any fold’s recommendation file is missing, the script will raise an error due to incomplete data.
>
> This step calculates the **10-fold average** and **standard deviation** for all evaluation metrics.

##### ▶️ Calculate the *p*-values and Cohen’s *d* values (CVH-REC vs. R2API)

After obtaining the **10-fold cross-validation results of CVH-REC**, follow the steps below to perform the statistical analysis:

1.  **Run the baseline model R2API** and generate its 10-fold recommendation results.

2.  Copy the resulting files of R2API into the  ../`output/` folder.

3.  Execute the following command to compute the statistical differences between the two methods:

        python statistical.py --method1 CVH-REC --method2 R2API

This script will produce the\*\* *p*-values\*\* and **Cohen’s *d* effect sizes**, which quantify the statistical significance and the magnitude of performance differences between CVH-REC and R2API.

***

#### 🔄 General Note for RQ2–RQ6

The **training and evaluation workflow** for subsequent research questions (RQ2-RQ6) is **identical to RQ1**.

For each variant, the process follows the same three parts:

1.  **Step 1 — Train the model and generate results** for each fold

2.  **Cross-Validation — Repeat Step 1** for all 10 folds (`fold1`–`fold10`)

3.  **Step 2 — Compute evaluation metrics** after all folds are completed

Below are specific commands and configurations for each RQ variant.

This step calculates the **10-fold average** and **standard deviation** for all evaluation metrics.

***

#### 🔍 RQ2: **Impact of sub-hypergraphs, behavioral similarity sub-hypergraphs, and semantic similarity subhypergraphs**

##### (1) **CVH-subHypergraphs**

▶️Step1. Train and generate results for a single fold (fold1):

    python main_global.py --dataset fold1 --patience 5 --epoch 200 --lr 0.002 --alpha1 0.6 --alpha2 0.6 --alpha3 0.6 --alpha4 0.6

    python main_text_encoder.py --dataset fold1 --method Global --patience 5 --lr 0.002 --epoch 200

🔁 Cross-Validation.&#x20;

Repeat for folds **2–10** by changing the `--dataset` parameter.

▶️Step 2. **Compute metrics**:

    python metrics.py --method Global

##### (2) **CVH-behavioralSimilarity**

▶️Step1. Train and generate results for a single fold (fold1):

    python main_without_call_view.py --dataset fold1 --patience 5 --epoch 200 --lr 0.002 --alpha1 0.6 --alpha2 0.6 --alpha3 0.6 --alpha4 0.6

    python main_text_encoder.py --dataset fold1 --method Without_call_view --patience 5 --lr 0.002 --epoch 200

🔁 Cross-Validation.

Repeat for folds **2–10** by changing the `--dataset` parameter.

▶️Step 2. **Compute metrics**:

    python metrics.py --method Without_call_view

##### (3) **CVH-semanticSimilarity**

▶️Step1. Train and generate results for a single fold (fold1):

    python main_without_tag_view.py --dataset fold1 --patience 5 --epoch 200 --lr 0.002 --alpha1 0.6 --alpha2 0.6 --alpha3 0.6 --alpha4 0.6

    python main_text_encoder.py --dataset fold1 --method Without_tag_view --patience 5 --lr 0.002 --epoch 200

🔁 Cross-Validation.

Repeat for folds **2–10** by changing the `--dataset` parameter.

▶️Step 2. **Compute metrics**:

    python metrics.py --method Without_tag_view

#### 🔍 **RQ3: Impact of contrastive learning, global contrastive learning, and local contrastive learning**

##### (1) **CVH-contrastive**

▶️Step1. Train and generate results for a single fold (fold1):

    python main_without_CL.py --dataset fold1 --patience 5 --epoch 200 --lr 0.002 --alpha1 0.6 --alpha2 0.6 --alpha3 0.6 --alpha4 0.6

    python main_text_encoder.py --dataset fold1 --method Without_CL --patience 5 --lr 0.002 --epoch 200

🔁 Cross-Validation.

Repeat for folds **2–10** by changing the `--dataset` parameter.

▶️Step 2. **Compute metrics**:

    python metrics.py --method Without_CL

##### (2) **CVH-globalContrastive**

▶️Step1. Train and generate results for a single fold (fold1):

    python main_without_global_CL.py --dataset fold1 --patience 5 --epoch 200 --lr 0.002 --alpha1 0.6 --alpha2 0.6 --alpha3 0.6 --alpha4 0.6

    python main_text_encoder.py --dataset fold1 --method Without_global_CL --patience 5 --lr 0.002 --epoch 200

🔁 Cross-Validation.

Repeat for folds **2–10** by changing the `--dataset` parameter.

▶️Step 2. **Compute metrics**:

    python metrics.py --method Without_global_CL

##### (3) **CVH-localContrastive**

▶️Step1. Train and generate results for a single fold (fold1):

    python main_without_local_CL.py --dataset fold1 --patience 5 --epoch 200 --lr 0.002 --alpha1 0.6 --alpha2 0.6 --alpha3 0.6 --alpha4 0.6

    python main_text_encoder.py --dataset fold1 --method Without_local_CL --patience 5 --lr 0.002 --epoch 200

🔁 Cross-Validation.

Repeat for folds **2–10** by changing the `--dataset` parameter.

▶️Step 2. **Compute metrics**:

    python metrics.py --method Without_local_CL

#### 🔍 **RQ4: Impact of multi-task learning**

##### (1) **CVH-multi-task**

▶️Step1. Train and generate results for a single fold (fold1):

    python main_without_multi.py --dataset fold1 --patience 5 --epoch 200 --lr 0.002 --alpha1 0.6 --alpha2 0.6 --alpha3 0.6 --alpha4 0.6

    python main_text_encoder.py --dataset fold1 --method Without_multi --patience 5 --lr 0.002 --epoch 200

🔁 Cross-Validation.

Repeat for folds **2–10** by changing the `--dataset` parameter.

▶️Step 2. **Compute metrics**:

    python metrics.py --method Without_multi

#### 🔍 RQ5: **Impact of requirement vector transfer**

##### (1) **CVH(semantic)**

▶️Step1. Train and generate results for a single fold (fold1):

    python main.py --dataset fold1 --patience 5 --epoch 200  --lr 0.002 --alpha1 0.6 --alpha2 0.6 --alpha3 0.6 --alpha4 0.6

    python main_semantic.py --dataset fold1

🔁 Cross-Validation.

Repeat for folds **2–10** by changing the `--dataset` parameter.

▶️Step 2. **Compute metrics**:

    python metrics.py --method CVH-REC_semantic

##### (2) **CVH(R2API)**

▶️Step1. Train and generate results for a single fold (fold1):

    python main.py --dataset fold1 --patience 5 --epoch 200 --lr 0.002 --alpha1 0.6 --alpha2 0.6 --alpha3 0.6 --alpha4 0.6

    python main_R2API.py --dataset fold1

🔁 Cross-Validation.

Repeat for folds **2–10** by changing the `--dataset` parameter.

▶️Step 2. **Compute metrics**:

    python metrics.py --method CVH-REC_R2API

#### 🔍 RQ6: **Impact of assigning the semantic vectors**

##### (1) **CHV-semantic**

▶️Step1. Train and generate results for a single fold (fold1):

    python main_random_init.py --dataset fold1 --patience 5 --epoch 200 --lr 0.002 --alpha1 0.6 --alpha2 0.6 --alpha3 0.6 --alpha4 0.6

    python main_text_encoder.py --dataset fold1 --method Random_init  --patience 5 --lr 0.002 --epoch 200

🔁 Cross-Validation.&#x20;

Repeat for folds **2–10** by changing the `--dataset` parameter.

▶️Step 2. **Compute metrics**:

    python metrics.py --method Random_init

## 3. Hyperparameter setting and implementation details

The hyperparameters of this project are defined in `utility/parser.py`. Below is the hyperparameter setting and implementation details.

| Variable name          | Description                             | Setting                                            |
| :--------------------- | :-------------------------------------- | :------------------------------------------------- |
| alpha1                 | Mashup behavioral similarity threshold  | 0.6                                                |
| alpha2                 | Mashup semantic similarity threshold    | 0.6                                                |
| alpha3                 | Web API behavioral similarity threshold | 0.6                                                |
| alpha4                 | Web API semantic similarity threshold   | 0.6                                                |
| beta                   | The weighting of contrastive learning   | 0.1                                                |
| regs                   | Weight decay                            | 1e-4                                               |
| lr                     | Inital learning rate                    | 0.002                                              |
| batch\_size            | Batch size                              | 512                                                |
| epoch                  | epoch                                   | 200                                                |
| patience               | patience of early stop                  | 5                                                  |
| seed                   |                                         | 85                                                 |
| Early stopping         |                                         | Loss no improvement for 5 epochs                   |
| Hardware               |                                         | NVIDIA RTX4090 24GB, withan 8124M Processor CPU\*2 |
| Framework              |                                         | PyTorch 2.7, CUDA 12.6                             |
| Learning rate schedule |                                         | Exponential decay schedule                         |
| Optimizer              |                                         | Adam                                               |

