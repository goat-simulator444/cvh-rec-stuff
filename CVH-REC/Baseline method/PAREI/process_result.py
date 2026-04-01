import os
import json
from collections import defaultdict


# 读取实际调用的API数据
def read_api_interaction(file_path):
    api_target_dict = defaultdict(list)
    with open(file_path, 'r') as f:
        for line in f:
            mashup_id, api_id = map(int, line.strip().split())
            api_target_dict[mashup_id].append(api_id)
    return api_target_dict


# 读取result目录下的推荐API数据并处理
def process_result_file(filename, result_dir):
    recommendation_dict = defaultdict(float)  # 用于存储单个文件的API得分

    # 读取单个文件
    with open(os.path.join(result_dir, filename), 'r') as f:
        for line in f:
            api_id, cosine_similarity, source_mashup_id, score = line.strip().split()
            api_id, cosine_similarity, source_mashup_id, score = int(api_id), float(cosine_similarity), int(
                source_mashup_id), float(score)
            # 计算得分并累加
            recommendation_dict[api_id] += cosine_similarity * score

    # 排序并选择得分最高的前20个API
    sorted_apis = sorted(recommendation_dict.items(), key=lambda x: x[1], reverse=True)[:20]
    recommended_apis = [api for api, score in sorted_apis]

    # 从文件名获取Mashup ID
    mashup_id = int(filename[4:].split('.')[0])  # 文件名格式为 demo+Mashup ID.txt
    return mashup_id, recommended_apis


# 合并推荐结果和实际调用API数据
def combine_recommendations_and_actuals(recommendations, api_target_dict):
    result = []
    for mashup_id, recommend_apis in recommendations.items():
        api_target = api_target_dict.get(mashup_id, [])
        # new_id=[i for i in range(5433,5483)]
        # if mashup_id not in new_id:
        mashup_id=mashup_id-1382
        # else:
        #     mashup_id=mashup_id-433
        result.append({
            "mashup_id": mashup_id,
            "recommend_api": recommend_apis,
            "api_target": api_target
        })
    return result


# 将结果写入JSON文件（每个Mashup占一行）
def write_to_json(data, output_file):
    with open(output_file, 'a') as f:  # 使用'a'模式以便逐行写入
        for entry in data:
            json.dump(entry, f, separators=(',', ':'))
            f.write('\n')  # 每个Mashup占一行


def main():
    # 设置文件路径
    result_dir = './result'  # 存放推荐结果文件的目录
    interaction_file = './data/MA_Interactive.txt'  # 存放实际调用数据的文件
    output_file = 'PAREI.json'  # 输出结果的JSON文件

    # 读取实际调用的API数据
    api_target_dict = read_api_interaction(interaction_file)

    # 清空输出文件
    open(output_file, 'w').close()

    # 提取demo文件并按数字编号排序
    demo_files = [
        filename for filename in os.listdir(result_dir) if filename.startswith('demo') and filename.endswith('.txt')
    ]
    # 按数字排序：demo940.txt → 940
    demo_files.sort(key=lambda name: int(name[4:].split('.')[0]))
    # 逐个处理每个demo文件

    # 再遍历已排序的文件
    for filename in demo_files:
        mashup_id, recommended_apis = process_result_file(filename, result_dir)
        recommendations = {mashup_id: recommended_apis}

        # 合并推荐结果和实际调用数据
        final_data = combine_recommendations_and_actuals(recommendations, api_target_dict)

        # 将结果写入JSON文件（每个Mashup占一行）
        write_to_json(final_data, output_file)
        # print(f"处理文件 {filename} 完成")

    print(f"所有文件处理完成，结果已保存到 {output_file}")


if __name__ == '__main__':
    main()
