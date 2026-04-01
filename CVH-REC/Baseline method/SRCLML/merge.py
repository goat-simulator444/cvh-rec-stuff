import json
import os

# 定义文件路径
file_paths = [f"./output/recommend_DNN_{i}.json" for i in range(10)]

# 用来存储合并后的数据
all_data = []

# 读取每个 JSON 文件并合并
for file_path in file_paths:
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                # 读取文件中的每个 JSON 对象
                file_data = f.read().splitlines()  # 按行读取文件
                for line in file_data:
                    all_data.append(json.loads(line))  # 解析每一行
            except json.JSONDecodeError as e:
                print(f"文件 {file_path} 解析失败: {e}")

# 按照 mashup_id 升序排序
all_data.sort(key=lambda x: x["mashup_id"])

# 保存合并后的数据到新的 JSON 文件
with open('./output/merged_recommend_data.json', 'w', encoding='utf-8') as f:
    for item in all_data:
        json.dump(item, f, ensure_ascii=False)
        f.write("\n")  # 每个 JSON 对象一行

print("合并后的数据已保存到 'merged_recommend_data.json'")
