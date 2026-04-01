import json
import os
import h5py
import numpy as np
from sentence_transformers import SentenceTransformer

# 加载数据
with open('Original Dataset/mashup_shuffled.json', 'r', encoding='utf-8') as f:
    mashups = json.load(f)
# import random
# random.shuffle(mashups)  # 原地打乱列表顺序

with open('Original Dataset/apis.json', 'r', encoding='utf-8') as f:
    apis = json.load(f)

# 构造API的id到tags映射
api_id_to_tags = {}
all_api_tags = set()
for api in apis:
    api_id = api['id']
    tags_str = api['details'].get('tags', '')
    tags = [t.strip() for t in tags_str.split(',') if t.strip()]
    api_id_to_tags[api_id] = tags
    all_api_tags.update(tags)

# 构造mashup的id到categories映射
mashup_id_to_tags = {}
all_mashup_tags = set()
for m in mashups:
    mid = m['id']
    cats = [c.strip() for c in m['categories'].split(',') if c.strip()]
    mashup_id_to_tags[mid] = cats
    all_mashup_tags.update(cats)

# 生成tag映射id
api_tag2id = {tag: i for i, tag in enumerate(sorted(all_api_tags))}
mashup_tag2id = {tag: i for i, tag in enumerate(sorted(all_mashup_tags))}



# 初始化SentenceTransformer模型
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')


def save_vectors_h5(filename, id_to_text):
    with h5py.File(filename, 'w') as f:
        for id_, text in id_to_text.items():
            vec = model.encode(text)
            f.create_dataset(str(id_), data=vec.astype(np.float32))

os.makedirs('data', exist_ok=True)
# 生成所有tag的向量
save_vectors_h5('data/api_tag_vector.h5', {tid: tag for tag, tid in api_tag2id.items()})
save_vectors_h5('data/mashup_tag_vector.h5', {tid: tag for tag, tid in mashup_tag2id.items()})

# 生成API description向量
api_id_to_desc = {api['id']: api['details']['description'] for api in apis}
save_vectors_h5('data/API_vectors.h5', api_id_to_desc)

# 生成mashup description向量
mashup_id_to_desc = {m['id']: m['description'] for m in mashups}
save_vectors_h5('data/vectors.h5', mashup_id_to_desc)

import os
from math import ceil

# === 1️⃣ 前90%用于交叉验证 ===
N = len(mashups)
cv_end = round(N * 0.9)     # 前90%做5折CV
cv_mashups = mashups[:cv_end]

# === 2️⃣ 后10%作为验证集 ===
val_mashups = mashups[cv_end:]

# === 3️⃣ 计算fold大小 ===
folds = 10
fold_size = ceil(len(cv_mashups) / folds)

# === 4️⃣ 创建5折数据集 ===
for i in range(folds):
    fold_dir = f'dataset/fold{i + 1}'
    os.makedirs(fold_dir, exist_ok=True)

    start = i * fold_size
    end = min((i + 1) * fold_size, len(cv_mashups))

    test_set = cv_mashups[start:end]
    train_set = cv_mashups[:start] + cv_mashups[end:]

    # 写训练集（TE.csv）
    with open(os.path.join(fold_dir, 'TE.csv'), 'w', encoding='utf-8') as f:
        for m in train_set:
            line = [str(m['id'])] + [str(api_id) for api_id in m.get('api_info', [])]
            f.write(' '.join(line) + '\n')

    # 写测试集（RS.csv）
    with open(os.path.join(fold_dir, 'RS.csv'), 'w', encoding='utf-8') as f:
        for m in test_set:
            line = [str(m['id'])] + [str(api_id) for api_id in m.get('api_info', [])]
            f.write(' '.join(line) + '\n')

    # Api_tag_mapping.csv
    with open(os.path.join(fold_dir, 'Api_tag_mapping.csv'), 'w', encoding='utf-8') as f:
        for tag, tid in api_tag2id.items():
            f.write(f'{tid}, {tag}\n')

    # api_tag.csv
    with open(os.path.join(fold_dir, 'api_tags.csv'), 'w', encoding='utf-8') as f:
        for api_id, tags in api_id_to_tags.items():
            tag_ids = [str(api_tag2id[t]) for t in tags if t in api_tag2id]
            f.write(f'{api_id} ' + ' '.join(tag_ids) + '\n')

    # mashup_tag_mapping.csv
    with open(os.path.join(fold_dir, 'mashup_tag_mapping.csv'), 'w', encoding='utf-8') as f:
        for tag, tid in mashup_tag2id.items():
            f.write(f'{tid}, {tag}\n')

    # # mashup_tag.csv
    # with open(os.path.join(fold_dir, 'mashup_tags.csv'), 'w', encoding='utf-8') as f:
    #     for m in mashups:
    #         tags = mashup_id_to_tags[m['id']]
    #         tag_ids = [str(mashup_tag2id[t]) for t in tags if t in mashup_tag2id]
    #         f.write(f"{m['id']} " + ' '.join(tag_ids) + '\n')

    # mashup_tag.csv（只写入训练集中的 mashup）
    with open(os.path.join(fold_dir, 'mashup_tags.csv'), 'w', encoding='utf-8') as f:
        for m in train_set:
            tags = mashup_id_to_tags.get(m['id'], [])
            tag_ids = [str(mashup_tag2id[t]) for t in tags if t in mashup_tag2id]
            f.write(f"{m['id']} " + ' '.join(tag_ids) + '\n')