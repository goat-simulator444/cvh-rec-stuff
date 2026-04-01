import pickle
import torch
import numpy as np
from tqdm import tqdm
from utils import get_indices
from transformers import BertTokenizer, BertModel
import pandas as pd
from bert_whitening import sents_to_vecs, transform_and_normalize, normalize
import json

dim = 256
USE_WHITENING = True
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('./data/mashup_name.json', 'r', encoding='utf-8') as file:
    mashups_ = json.load(file)

X_train, X_test, oov = get_indices()

with open('./data/mashup_description.json', 'r', encoding='utf-8') as f:
    mashup_description_ = json.load(f)

with open('./data/mashup_used_api.json', 'r', encoding='utf-8') as f:
    mashup_apis_ = json.load(f)

with open('./data/mashup_category.json', 'r', encoding='utf-8') as f:
    mashup_category_ = json.load(f)


tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
model = BertModel.from_pretrained("bert-base-cased")
model.to(DEVICE)

# Prepare train/test split
filter_idx = []
test_mashup_list = []
train_code_list, train_api_list, train_cate_list = [], [], []
test_code_list, test_api_list, test_cate_list = [], [], []

for idx, mashup in enumerate(mashups_):
    if mashup in X_train:
        filter_idx.append(idx)
    else:
        test_mashup_list.append(mashup)

for idx, desc in enumerate(mashup_description_):
    desc_text = ' '.join(desc).strip().rstrip()
    if idx in filter_idx:
        train_code_list.append(desc_text)
        train_api_list.append(mashup_apis_[idx])
        train_cate_list.append(mashup_category_[idx])
    else:
        test_code_list.append(desc_text)
        test_api_list.append(mashup_apis_[idx])
        test_cate_list.append(mashup_category_[idx])

def sim_jaccard(s1, s2):
    s1, s2 = set(s1), set(s2)
    return len(s1 & s2) / len(s1 | s2)

class Retrieval:
    def __init__(self):
        with open('./model/code_vector_whitening.pkl', 'rb') as f:
            self.bert_vec = pickle.load(f)
        with open('./model/kernel.pkl', 'rb') as f:
            self.kernel = pickle.load(f)
        with open('./model/bias.pkl', 'rb') as f:
            self.bias = pickle.load(f)
        self.vecs = None
        self.id2text = {}

    def encode_file(self):
        self.vecs = np.array([vec.reshape(1, -1) for vec in self.bert_vec], dtype='float32').squeeze(1)
        self.id2text = {i: train_code_list[i] for i in range(len(train_code_list))}

    def cosine_similarity(self, vec1, vecs):
        dot = np.dot(vecs, vec1.T).squeeze()
        norm_vecs = np.linalg.norm(vecs, axis=1)
        norm_vec1 = np.linalg.norm(vec1)
        return dot / (norm_vecs * norm_vec1 + 1e-8)

    def single_query(self, code, ast, topK):
        vec = sents_to_vecs([code], tokenizer, model)
        if USE_WHITENING:
            vec = transform_and_normalize(vec, self.kernel, self.bias)
        else:
            vec = normalize(vec)[:, :dim]
        sim_scores = self.cosine_similarity(vec, self.vecs)
        top_indices = np.argsort(sim_scores)[::-1][:topK]
        top_scores = sim_scores[top_indices]
        return top_indices.tolist(), top_scores.tolist()

if __name__ == '__main__':
    retriever = Retrieval()
    retriever.encode_file()

    sim_nl_list, sim_api_list = [], []

    sim_nl_list, sim_api_list = [], []
    top5_mashup_ids = []
    top5_sim_scores = []

    top5_mashup_ids, top5_scores = [], []
    for i in tqdm(range(len(train_code_list))):
        sim_code, sim_dis = retriever.single_query(train_code_list[i], train_code_list[i], topK=5)
        top5_mashup_ids.append(sim_code)  # 保留索引而不是ID
        top5_scores.append(sim_dis)

    pd.DataFrame(top5_mashup_ids).to_csv("./data/Train_Top5_MashupIDs.csv", index=False, header=None)
    pd.DataFrame(top5_scores).to_csv("./data/Train_Top5_Scores.csv", index=False, header=None)

    for i in tqdm(range(len(train_code_list))):
        sim_code, sim_dis = retriever.single_query(train_code_list[i], train_code_list[i], topK=5)
        sim_nl_list.append([sim_code[1]])
        sim_api_list.append([sim_dis[1]])

    pd.DataFrame(sim_nl_list).to_csv("./data/Semantic_train.csv", index=False, header=None)
    df = pd.DataFrame(sim_api_list)
    df = (df - df.min()) / (df.max() - df.min())
    df.to_csv("./data/Semantic_train_api.csv", index=False, header=None)

    sim_nl_list, sim_api_list = [], []
    for i in tqdm(range(len(test_code_list))):
        sim_code, sim_dis = retriever.single_query(test_code_list[i], test_code_list[i], topK=5)
        sim_nl_list.append(sim_code[0])
        sim_api_list.append(sim_dis[0])
        all_mashup_services = train_api_list[sim_code[0]]
        if test_mashup_list[i] in ['soundpushr', 'explore-travellr', 'gregs-alerts']:
            print(test_mashup_list[i])
            print(test_api_list[i])
            print(test_code_list[i])
            print(test_cate_list[i])
            print(all_mashup_services)
            print(train_code_list[sim_code[0]])
            print(train_cate_list[sim_code[0]])

    pd.DataFrame(sim_nl_list).to_csv("./data/Semantic_test.csv", index=False, header=None)
    df = pd.DataFrame(sim_api_list)
    df = (df - df.mean()) / df.std()
    df = (df - df.min()) / (df.max() - df.min())
    df.to_csv("./data/Semantic_test_api.csv", index=False, header=None)


    # 将csv文件复制到data_api文件夹下
    import shutil
    import glob
    import os

    # 创建目标文件夹（如果不存在）
    os.makedirs("./data_api", exist_ok=True)

    # 获取所有 data 文件夹下的 csv 文件
    csv_files = glob.glob("./data/*.csv")

    # 复制每个文件到 data_api 文件夹
    for file in csv_files:
        shutil.copy(file, "./data_api")
