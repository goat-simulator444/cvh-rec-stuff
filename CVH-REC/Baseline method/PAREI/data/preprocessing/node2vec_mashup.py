import networkx as nx
from node2vec import Node2Vec
import numpy as np
import os

# 获取当前文件的目录和数据目录
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.dirname(current_dir)  # 父目录
# 加载 Mashup 描述（）
description_text = []  # 存储每个 Mashup 的描述文本
def load_mashup_descriptions(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            description_text.append(line.split(' =->= ')[1].replace("\n", ''))

# 加载 Mashup 和 API 之间的调用关系
mashup_api_relations = {}  # 存储 Mashup 和 API 的调用关系
def load_mashup_api_relations(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            mashup_id = int(line.split()[0])
            api_id = int(line.split()[1])
            if mashup_id not in mashup_api_relations:
                mashup_api_relations[mashup_id] = []
            mashup_api_relations[mashup_id].append(api_id)

# 构建图
def build_graph():
    G = nx.Graph()

    # 为每个 Mashup 和 API 添加节点
    for mashup_id in mashup_api_relations:
        G.add_node(mashup_id, type='mashup')  # 添加 Mashup 节点
        for api_id in mashup_api_relations[mashup_id]:
            G.add_node(api_id, type='api')  # 添加 API 节点
            G.add_edge(mashup_id, api_id)  # 添加边，表示 Mashup 调用 API

    return G

# 使用 Node2Vec 生成嵌入向量
def generate_node2vec_embeddings(G, dimensions=100, walk_length=5):
    node2vec = Node2Vec(G, dimensions=dimensions, walk_length=walk_length)
    model = node2vec.fit()

    return model

# 提取 Mashup 节点的嵌入向量
def get_mashup_embeddings(model, mashup_ids):
    mashup_embeddings = {}
    for mashup_id in mashup_ids:
        mashup_embeddings[mashup_id] = model.wv[str(mashup_id)]  # 获取 Mashup 的嵌入向量
    return mashup_embeddings

# 保存嵌入向量到文件
def save_embeddings_to_file(embeddings, filename='data/mashup_node2vec_embeddings.txt'):
    with open(filename, 'w') as file:
        file.write(f"{len(embeddings.items())} {100}\n")
        for node_id, emb in embeddings.items():
            emb_str = ' '.join(map(str, emb))
            file.write(f"{node_id} {emb_str}\n")

# 主函数
if __name__ == "__main__":
    # 加载数据
    load_mashup_descriptions('data/Mashup_description.txt')
    load_mashup_api_relations('data/MA_Interactive.txt')

    # 构建图
    G = build_graph()

    # 使用 Node2Vec 生成嵌入向量
    print("正在训练 Node2Vec 模型...")
    model = generate_node2vec_embeddings(G)

    # 获取所有 Mashup 节点的嵌入向量
    print("正在提取 Mashup 节点的嵌入向量...")
    mashup_embeddings = get_mashup_embeddings(model, mashup_api_relations.keys())

    # 保存嵌入向量到文件
    print("正在保存嵌入向量到文件...")
    save_embeddings_to_file(mashup_embeddings)

    print("嵌入向量生成并保存完成。")
