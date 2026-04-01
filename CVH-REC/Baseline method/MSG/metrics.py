import json
def compute_recall_at_n(filename='test_results.json', N=5):
    with open(filename, 'r') as f:
        all_results = json.load(f)

    total_recall = 0
    count = 0

    for result in all_results:
        true_apis = set(result['true_apis'])
        # predicted_apis = result['predicted_apis'][:N]  # 取前 N 个预测 API
        predicted_apis = list(dict.fromkeys(result['predicted_apis']))[:N]
        print(predicted_apis)

        # 计算 recall@N
        intersection = true_apis.intersection(predicted_apis)
        recall_at_n = len(intersection) / len(true_apis) if true_apis else 0

        total_recall += recall_at_n
        count += 1

    # 返回平均 recall@N
    return total_recall / count if count > 0 else 0


# 调用示例
recall_at_5 = compute_recall_at_n(N=10)
print(f"{recall_at_5:.4f}")
