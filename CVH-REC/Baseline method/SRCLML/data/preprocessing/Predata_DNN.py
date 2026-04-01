import json
from collections import defaultdict

# 读取JSON文件
def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# 处理数据并生成四个文件
def process_data(data):
    # 文件1: descriptions.txt
    descriptions = []
    # 文件2: categories.txt
    categories_map = {}
    categories_list = defaultdict(list)
    # 文件3: categories_map.txt
    category_index = 0
    # 文件4: api_calls.txt
    api_calls = []

    for item in data:
        mashup_id = item['id']
        description = item['description']
        description = description.replace("\n", " ").strip()  # 去掉多余的空白字符
        categories = item['categories'].split(',')
        api_info = item['api_info']

        # 文件1: descriptions.txt
        descriptions.append(f"{mashup_id},{description}")

        # 文件2: categories.txt
        category_indices = []
        for category in categories:
            category = category.strip()
            if category not in categories_map:
                categories_map[category] = category_index
                category_index += 1
            category_indices.append(categories_map[category])
        categories_list[mashup_id].extend(category_indices)

        # 文件4: api_calls.txt
        api_calls.append(f"{mashup_id} {' '.join(map(str, api_info))}")

    # 生成文件3: categories_map.txt
    categories_map_lines = [f"{index},{category}" for category, index in categories_map.items()]

    return descriptions, categories_list, categories_map_lines, api_calls

# 写入文件
def write_to_file(file_name, content):
    with open(file_name, 'w', encoding='utf-8') as file:
        file.write('\n'.join(content))

# 主函数
def main():
    json_file_path = './data/numbered_filtered_mashup_details_new.json'  # 替换为你的JSON文件路径
    data = read_json_file(json_file_path)

    descriptions, categories_list, categories_map_lines, api_calls = process_data(data)

    # 写入文件1: descriptions.txt
    write_to_file('./data/descriptions.txt', descriptions)
    # 写入文件2: categories.txt
    categories_lines = [f"{mashup_id} {' '.join(map(str, indices))}" for mashup_id, indices in sorted(categories_list.items())]
    write_to_file('./data/categories.txt', categories_lines)
    # 写入文件3: categories_map.txt
    write_to_file('./data/categories_map.txt', categories_map_lines)
    # 写入文件4: api_calls.txt
    write_to_file('./data/api_calls.txt', api_calls)

    print("所有文件已生成！")

if __name__ == "__main__":
    main()