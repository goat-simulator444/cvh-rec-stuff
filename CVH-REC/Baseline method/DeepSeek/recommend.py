import time
import argparse
import requests
import ast
import json


def transform_url_id(api_list, api_datas):
    api_id_list = []
    for api in api_list:
        api_id = -1
        for api_data in api_datas:
            if api_data['url'] == api:
                if api_data['id'] not in api_id_list:
                    api_id = api_data['id']
            elif api in api_data['url']:
                if api_id == -1:
                    if api_data['id'] not in api_id_list:
                        api_id = api_data['id']
            elif api_data['url'] in api:
                if api_id == -1:
                    if api_data['id'] not in api_id_list:
                        api_id = api_data['id']
        api_id_list.append(api_id)
    return api_id_list


def generate_recommendation(str):
    url = "https://api.siliconflow.cn/v1/chat/completions"
    payload = {
        "model": "deepseek-ai/DeepSeek-V3",
        "messages": [
            {
                "role": "user",
                "content": str
            }
        ],
        "max_tokens": 1000,
        "temperature": 0.1
    }
    headers = {
        "Authorization": key,
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)

    # 提取并打印文本内容
    if response.status_code == 200:
        response_json = response.json()
        message_content = response_json['choices'][0]['message']['content']
        return message_content
    else:
        print("Error:", response.text)
        return None


def extract_api_names(mapping_file):
    with open(mapping_file, 'r', encoding='utf-8') as f:
        api_data = json.load(f)

    # 提取每个 API 的 url 字段（位于顶层）
    api_urls = [entry['url'].strip() for entry in api_data if 'url' in entry]
    return api_urls


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Mashup API Recommendation Script")
    parser.add_argument('--key', type=str, required=True, help="Your API Authorization key")
    args = parser.parse_args()

    key = args.key  # 获取命令行输入的 key
    mapping_path = 'api_id_mapping.json'
    api_url_list = extract_api_names(mapping_path)
    with open('api_id_mapping.json', 'r', encoding='utf-8') as f1:
        api_datas = json.load(f1)

    output_file = open('result.json', 'a', encoding='utf-8')
    with open('shuffle_mashup_details.json', 'r', encoding='utf-8') as f:
        mashup_datas = json.load(f)

    for id in range(0, 51):
        time.sleep(60)
        description = mashup_datas[id]['description']
        print(mashup_datas[id]['id'])
        str1 = (f"The candidate API list is {api_url_list}. I want to complete a mashup, and the requirement is as follow:: {description}. "
                f"Please recommend 20 suitable web APIs based on my requirement."
                f"Output the recommended web API as a python list (e.g., ['api1', 'api2', ..., 'api20'])."
                f"Do not output anything other than the recommended web APIs.")

        r = generate_recommendation(str1)
        while True:
            try:
                import re
                match = re.search(r'\[.*\]', r)

                if match:
                    # 提取方括号及其中的内容
                    r = match.group(0)
                rec_list = ast.literal_eval(r)
                print("Successfully converted to list:", rec_list)
                break

            except (ValueError, SyntaxError) as e:
                print(f"Failed to convert: {r} ({e})")
                r = generate_recommendation(str1)

        api_id_list = transform_url_id(rec_list, api_datas)

        mashup_id = mashup_datas[id]['id']
        recommend_api = api_id_list + [-1] * (20 - len(api_id_list))
        remove_apis = mashup_datas[id]['api_info']
        data = {
            "mashup_id": mashup_id,
            "recommend_api": recommend_api,
            "api_target": remove_apis
        }
        json_str = json.dumps(data, ensure_ascii=False)
        print(json_str)

        output_file.write(json_str + '\n')
        output_file.flush()