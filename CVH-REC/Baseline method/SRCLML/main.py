import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
import os
import ast
import json

# 设置 SOCKS5 代理
# os.environ['http_proxy'] = 'socks5://127.0.0.1:10808'
# os.environ['https_proxy'] = 'socks5://127.0.0.1:10808'


# 读取文件并准备数据
def read_mashup_descriptions(file_path):
    descriptions = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(",", 1)  # ID 和 描述部分
            descriptions.append(parts[1])  # 只保存描述
    return descriptions


def read_labels(file_path):
    labels = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            mashup_id = int(parts[0])
            labels[mashup_id] = list(map(int, parts[1:]))  # 标签为一个列表
    return labels


def read_api_calls(file_path):
    api_calls = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            mashup_id = int(parts[0])
            api_ids = list(map(int, parts[1:]))  # API ID 列表
            api_calls[mashup_id] = api_ids
    return api_calls


def read_fold_ids(file_path):
    # 读取每一折的Mashup ID
    folds = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            fold = ast.literal_eval(line.strip())  # 将字符串格式的列表转化为实际的列表
            folds.append(fold)
    return folds


# 使用Sentence-BERT模型进行文本向量化
def get_text_embeddings(descriptions, model_name='paraphrase-MiniLM-L6-v2'):
    # 加载Sentence-BERT模型
    model = SentenceTransformer(model_name)
    embeddings = model.encode(descriptions)
    return torch.tensor(embeddings, dtype=torch.float32)


# 创建三层的全连接DNN模型
class MultiTaskModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_api, num_tags):
        super(MultiTaskModel, self).__init__()

        # 三层全连接层
        self.input_fc = nn.Linear(input_size, hidden_size)
        self.hidden_fc1 = nn.Linear(hidden_size, hidden_size)
        self.hidden_fc2 = nn.Linear(hidden_size, hidden_size)
        self.api_fc = nn.Linear(hidden_size, num_api)  # API推荐任务
        self.tag_fc = nn.Linear(hidden_size, num_tags)  # 标签预测任务

    def forward(self, x):
        hidden = F.relu(self.input_fc(x))  # 第一层
        hidden = F.relu(self.hidden_fc1(hidden))  # 第二层
        hidden = F.relu(self.hidden_fc2(hidden))  # 第三层

        api_output = self.api_fc(hidden)  # API推荐输出
        tag_output = self.tag_fc(hidden)  # 标签预测输出

        # 进行Softmax激活，转换为概率分布
        # api_output = F.softmax(api_output, dim=-1)  # 对API推荐输出使用Softmax
        # tag_output = F.softmax(tag_output, dim=-1)  # 对标签预测输出使用Softmax

        return api_output, tag_output

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#训练和评估函数
def train_model(optimizer, model, train_data, api_labels, tag_labels, api_weight=1.0, tag_weight=1.0):
    model.train()
    optimizer.zero_grad()

    api_output, tag_output = model(train_data)

    # 计算API推荐部分的损失
    # api_loss = F.cross_entropy(api_output, api_labels)
    # # 计算标签预测部分的损失
    # tag_loss = F.cross_entropy(tag_output, tag_labels)

    api_loss = F.binary_cross_entropy_with_logits(api_output, api_labels)  # 使用logits而非softmax
    tag_loss = F.binary_cross_entropy_with_logits(tag_output, tag_labels)  # 使用logits而非softmax

    # 总损失 = 加权的API Loss + 标签预测 Loss
    total_loss = api_weight * api_loss + tag_weight * tag_loss

    total_loss.backward()
    optimizer.step()

    return total_loss.item(),api_loss,tag_loss


def evaluate_model(model, val_data, api_labels, tag_labels):
    model.eval()
    with torch.no_grad():
        api_output, tag_output = model(val_data)
        # api_loss = F.cross_entropy(api_output, api_labels)
        # tag_loss = F.cross_entropy(tag_output, tag_labels)
        # 使用BCEWithLogitsLoss来处理多标签情况
        api_loss = F.binary_cross_entropy_with_logits(api_output, api_labels)  # 使用logits而非softmax
        tag_loss = F.binary_cross_entropy_with_logits(tag_output, tag_labels)  # 使用logits而非softmax

    return api_loss.item(), tag_loss.item(),api_output,tag_output


# 读取数据
def preprocess_data(descriptions, api_calls, labels, input_size=384):  # BERT输出维度是384
    data = []  # 描述的向量表示
    api_labels = []  # API
    tag_labels = []  # Tag

    # 使用Sentence-BERT模型得到文本的向量表示
    description_vec = get_text_embeddings(descriptions)  # Sentence-BERT向量化

    for mashup_id, description in enumerate(descriptions):
        # 获取API标签
        api_tag = np.zeros(940)  # API ID最大为940
        if mashup_id in api_calls:
            for api_id in api_calls[mashup_id]:
                api_tag[api_id] = 1
        api_labels.append(api_tag)

        tag_tag = np.zeros(399)  # 标签数量为399，编号从0到398
        if mashup_id in labels:
            for tag in labels[mashup_id]:
                tag_tag[tag] = 1
        tag_labels.append(tag_tag)

    data = torch.tensor(np.array(description_vec), dtype=torch.float32).to(device)
    api_labels = torch.tensor(np.array(api_labels), dtype=torch.float32).to(device)
    tag_labels = torch.tensor(np.array(tag_labels), dtype=torch.float32).to(device)

    return data, api_labels, tag_labels

# 手动打乱数据函数
import random
def shuffle_data(data, api_labels, tag_labels):
    # 将数据、API标签和Tag标签组合成一个元组列表
    combined = list(zip(data, api_labels, tag_labels))
    # 打乱列表
    random.shuffle(combined)
    # 解包为数据、API标签和Tag标签
    shuffled_data, shuffled_api_labels, shuffled_tag_labels = zip(*combined)
    # 转换回torch.tensor
    shuffled_data = torch.stack(shuffled_data)
    shuffled_api_labels = torch.stack(shuffled_api_labels)
    shuffled_tag_labels = torch.stack(shuffled_tag_labels)
    return shuffled_data, shuffled_api_labels, shuffled_tag_labels

# 训练与十折交叉验证
def run_comparison_experiment(descriptions_file, api_calls_file, labels_file, folds_file, input_size=384,
                              hidden_size=128,
                              num_api=940, num_tags=399, num_epochs=1000, output_file="output_results_10_fold.txt"):
    descriptions = read_mashup_descriptions(descriptions_file)
    labels = read_labels(labels_file)
    api_calls = read_api_calls(api_calls_file)
    fold_ids = read_fold_ids(folds_file)  # 读取十折文件

    # 预处理数据
    data, api_labels, tag_labels = preprocess_data(descriptions, api_calls, labels, input_size)

    api_losses = []
    tag_losses = []

    # 将输出写入文件
    with open(output_file, 'w',encoding='GBK') as f:
        f.write("Training and evaluation results:\n")

        # 十折交叉验证
        for fold in range(len(fold_ids)):
            f.write(f"Fold {fold + 1}\n")
            print(f"Fold {fold + 1}")

            # 划分训练集和验证集
            val_ids = fold_ids[fold]
            train_ids = [i for i in range(len(data)) if i not in val_ids]

            train_data = data[train_ids]
            val_data = data[val_ids]
            train_api_labels = api_labels[train_ids]
            val_api_labels = api_labels[val_ids]
            train_tag_labels = tag_labels[train_ids]
            val_tag_labels = tag_labels[val_ids]

            # 初始化模型
            model = MultiTaskModel(input_size=input_size, hidden_size=hidden_size, num_api=num_api,
                                   num_tags=num_tags).to(device)

            # 训练模型
            optimizer = optim.Adam(model.parameters(), lr=0.01)
            for epoch in range(num_epochs):
                # 手动对train_data进行打乱
                train_data, train_api_labels, train_tag_labels = shuffle_data(train_data, train_api_labels,
                                                                              train_tag_labels)
                train_loss,api_loss,tag_loss = train_model(optimizer, model, train_data, train_api_labels, train_tag_labels)
                f.write(f"Epoch {epoch + 1} - 训练损失：{train_loss}\n")
                f.write(f"api损失：{api_loss}，tag损失：{tag_loss}\n")

            # 评估模型
            api_loss, tag_loss,api_output,tag_output = evaluate_model(model, val_data, val_api_labels, val_tag_labels)
            api_losses.append(api_loss)
            tag_losses.append(tag_loss)
            f.write(f"API推荐损失：{api_loss}\n")
            f.write(f"标签预测损失：{tag_loss}\n")

            # 保存推荐结果到JSON文件
            recommend_results = []
            predicted_apis = torch.argsort(api_output, dim=-1, descending=True)[:, :20].cpu().numpy()  # Top-20 推荐的API

            for idx, mashup_id in enumerate(val_ids):
                # new_id=[4493,4494,4495,4496,4497]
                # new_id = [i for i in range(4493, 4543)]
                # api=api_calls.get(mashup_id, [])
                # if mashup_id in new_id:
                #     mashup_id+=507
                result = {
                    "mashup_id": mashup_id,
                    "recommend_api": predicted_apis[idx].tolist(),
                    "api_target": api
                }
                recommend_results.append(result)

            # 保存每一折的推荐结果
            with open(f"./output/recommend_DNN_{fold}.json", 'w', encoding='utf-8') as file:
                for result in recommend_results:
                    json.dump(result, file, ensure_ascii=False)
                    file.write("\n")

        # 输出平均损失
        f.write(f"\n平均API推荐损失：{np.mean(api_losses)}\n")
        f.write(f"平均标签预测损失：{np.mean(tag_losses)}\n")
        print(f"\n平均API推荐损失：{np.mean(api_losses)}")
        print(f"平均标签预测损失：{np.mean(tag_losses)}")


# 运行对比实验
descriptions_file = './data/descriptions.txt'
api_calls_file = './data/api_calls.txt'
labels_file = './data/categories.txt'
folds_file = './data/10_fold_new.txt'
output_file = './output/results_10_fold.txt'
import os
if not os.path.exists('./output'):
    os.makedirs('./output')

run_comparison_experiment(descriptions_file, api_calls_file, labels_file, folds_file, output_file=output_file)
