import os
import json
import pickle
import numpy as np
from tqdm import tqdm

import torch
from transformers import BertTokenizer, BertModel

# 输入文件路径
mashup_path = 'Original data/programmable_mashups.json'
api_path = 'Original data/api.json'

# 输出目录
base_dir = 'data/api_mashup'
embedding_dir = os.path.join(base_dir, 'embeddings')
partial_api_dir = os.path.join(embedding_dir, 'partial')
os.makedirs(partial_api_dir, exist_ok=True)

# 加载数据
with open(mashup_path, 'r', encoding='utf-8') as f:
    mashups = json.load(f)
print(f'Loaded {len(mashups)} mashups.')
# 按照 id 从小到大排序 mashups
mashups.sort(key=lambda m: m.get("id", 0))

with open(api_path, 'r', encoding='utf-8') as f:
    apis = json.load(f)
print(f'Loaded {len(apis)} APIs.')

# 设备选择
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载BERT模型和tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.to(device)
model.eval()

def encode_texts(texts, batch_size=64):
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding texts"):
            batch_texts = texts[i:i+batch_size]
            encoded_input = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
            encoded_input = {k:v.to(device) for k,v in encoded_input.items()}
            output = model(**encoded_input)
            # 取[CLS] token的隐藏状态作为句子向量
            batch_embeddings = output.last_hidden_state[:,0,:].cpu().numpy()
            embeddings.append(batch_embeddings)
    return np.vstack(embeddings)

# 生成 mashup 描述文本列表
mashup_texts = [m.get("description", "").strip() or " " for m in mashups]
# 生成 api 描述文本列表
api_texts = [a.get("details", {}).get("description", "").strip() or " " for a in apis]

print("开始编码 Mashup 文本...")
mashup_embeddings = encode_texts(mashup_texts)

print("开始编码 API 文本...")
api_embeddings = encode_texts(api_texts)

# 保存 npy 文件
mashup_embedding_path = os.path.join(embedding_dir, 'text_bert_mashup_embeddings.npy')
np.save(mashup_embedding_path, mashup_embeddings)
print(f"✅ Saved {mashup_embedding_path}")

api_embedding_path = os.path.join(partial_api_dir, 'text_bert_api_embeddings.npy')
np.save(api_embedding_path, api_embeddings)
print(f"✅ Saved {api_embedding_path}")



import pandas as pd

# 构造一个 DataFrame，假设每个 mashup 的 id 和对应的 api_info 列表
invocation_df = pd.DataFrame([
    {"X": m.get("id"), "Y": m.get("api_info", [])} for m in mashups
])

# 保存为 pickle
invocation_df.to_pickle(os.path.join(base_dir, 'partial_invocation.pkl'))
print("✅ Saved partial_invocation.pkl as DataFrame")
