import json
import csv
import os

# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 构建完整的文件路径
api_file = os.path.join(current_dir, 'apis.json')
#api_file = "api_id_mapping.json"
mashup_file = os.path.join(current_dir, 'mashup_reindexed.json')

# ==== 读取 JSON 文件 ====
with open(api_file, 'r', encoding='utf-8') as f1:
    api_data = json.load(f1)

with open(mashup_file, 'r', encoding='utf-8') as f2:
    mashup_data = json.load(f2)

# ==== 获取api文件的最大ID+1 ,作为mashup文件的起始编号====
base_id = len(api_data)

# ==== 文件1 + 文件2 全部描述写入 file1_descriptions.txt ====
file1_output = []
for item in api_data:
    desc = item.get("details", {}).get("description", "").replace("\n", " ").strip()
    if desc =="":
        desc="API"
    file1_output.append(f'{item["id"]} =->= {desc}')

file2_output = []
interaction_output = []

for item in mashup_data:
    old_id = item["id"]
    new_id = old_id + base_id
    item["id"] = new_id  # 更新 ID

    # 2. 提取描述
    desc = item.get("description", "").replace("\n", " ").strip()
    desc_line = f"{new_id} =->= {desc}"
    file1_output.append(desc_line)
    file2_output.append(desc_line)

    # 3. 交互信息：每个 API 写一行
    for api_id in item.get("api_info", []):
        interaction_output.append(f"{new_id} {api_id}")

# 获取data目录的路径（当前目录的父目录）
data_dir = os.path.dirname(current_dir)
# 输出文件路径
data_description_file = os.path.join(data_dir, 'data_description.txt')
mashup_description_file = os.path.join(data_dir, 'Mashup_description.txt')
ma_interactive_file = os.path.join(data_dir, 'MA_Interactive.txt')
# ==== 写入 data_description.txt ====
with open(data_description_file, "w", encoding="utf-8", newline='') as f:
    for line in file1_output:
        f.write(line + "\n")

# ==== 写入 Mashup_description.txt ====
with open(mashup_description_file, "w", encoding="utf-8", newline='') as f:
    for line in file2_output:
        f.write(line + "\n")

# ==== 写入 MA_Interactive.txt ====
with open(ma_interactive_file, "w", encoding="utf-8", newline='') as f:
    for line in interaction_output:
        f.write(line + "\n")
