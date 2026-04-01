import os
import json
import math
from utility.parser import parse_args

args = parse_args()
Ns = [3, 5, 10, 20]  # 支持多个 N 值

# 加载 API ID -> 名称 映射
with open("Original Dataset/api.json", "r", encoding="utf-8") as f:
    api_data = json.load(f)
    api_id_to_title = {item["id"]: item["url"] for item in api_data}
    all_api_ids = set(api_id_to_title.keys())

# 用于累计所有 fold 的平均值（除了 coverage）
fold_results = {N: {"precision": [], "recall": [], "map": [], "ndcg": []} for N in Ns}

# 用于累计所有 fold 中推荐到的 API（用于最终的 coverage 计算）
total_recommended_api_set = {N: set() for N in Ns}

# 遍历存在的 fold 文件夹
for fold_idx in range(1, 11):
    fold_name = f"fold_{fold_idx}"
    rs_path = f"dataset/{fold_name}/RS.csv"
    rec_path = f"output/recommend_{args.similar_threthold}_{args.alpha1}_{args.alpha2}_{args.alpha3}_{args.alpha4}_{fold_name}.json"

    if not (os.path.exists(rs_path) and os.path.exists(rec_path)):
        print(f"Skipping {fold_name}, RS.csv or recommendation file not found.")
        continue

    # 加载 ground truth
    ground_truth = {}
    with open(rs_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            mashup_id = int(parts[0])
            apis = list(map(int, parts[1:]))
            ground_truth[mashup_id] = apis

    # 加载推荐结果
    recommendations = {}
    with open(rec_path, "r") as f:
        for line in f:
            data = json.loads(line)
            mashup_id = data["mashup_id"]
            recommend_api = data["recommend_api"]
            recommendations[mashup_id] = recommend_api

    # print(f"\nThe performance of {fold_name} is as follows:")

    for N in Ns:
        precision_list = []
        recall_list = []
        map_list = []
        ndcg_list = []
        recommended_api_set = set()  # 当前 fold 的推荐 API

        for mashup_id, true_apis in ground_truth.items():
            recommended_apis = recommendations.get(mashup_id, [])[:N]
            if not true_apis:
                continue

            true_set = set(true_apis)
            hit_count = len(set(recommended_apis) & true_set)
            recommended_api_set.update(recommended_apis)

            # Precision@N
            precision = hit_count / N
            precision_list.append(precision)

            # Recall@N
            recall = hit_count / len(true_apis)
            recall_list.append(recall)

            # MAP@N
            cor_list = [1.0 if recommended_apis[i] in true_set else 0.0 for i in range(len(recommended_apis))]
            sum_cor_list = sum(cor_list)
            if sum_cor_list == 0:
                map_score = 0.0
            else:
                summary = sum(
                    sum(cor_list[:i + 1]) / (i + 1) * cor_list[i]
                    for i in range(len(cor_list))
                )
                map_score = summary / sum_cor_list
            map_list.append(map_score)

            # NDCG@N
            dcg = sum(
                1 / math.log2(i + 2) if i < len(recommended_apis) and recommended_apis[i] in true_set else 0
                for i in range(N)
            )
            idcg = sum(1 / math.log2(i + 2) for i in range(min(len(true_apis), N)))
            ndcg = dcg / idcg if idcg != 0 else 0
            ndcg_list.append(ndcg)

        # 平均值
        average_precision = sum(precision_list) / len(precision_list) if precision_list else 0
        average_recall = sum(recall_list) / len(recall_list) if recall_list else 0
        average_map = sum(map_list) / len(map_list) if map_list else 0
        average_ndcg = sum(ndcg_list) / len(ndcg_list) if ndcg_list else 0

        # 更新累计推荐的 API 集合（用于最终 coverage）
        total_recommended_api_set[N].update(recommended_api_set)

        # 添加到整体结果中（不再保存 coverage）
        fold_results[N]["precision"].append(average_precision)
        fold_results[N]["recall"].append(average_recall)
        fold_results[N]["map"].append(average_map)
        fold_results[N]["ndcg"].append(average_ndcg)

# 打印整体平均结果（包含最终统一计算的 coverage）
print("\n==== Overall average across all folds ====")
for N in Ns:
    avg_prec = sum(fold_results[N]["precision"]) / len(fold_results[N]["precision"]) if fold_results[N]["precision"] else 0
    avg_recall = sum(fold_results[N]["recall"]) / len(fold_results[N]["recall"]) if fold_results[N]["recall"] else 0
    avg_map = sum(fold_results[N]["map"]) / len(fold_results[N]["map"]) if fold_results[N]["map"] else 0
    avg_ndcg = sum(fold_results[N]["ndcg"]) / len(fold_results[N]["ndcg"]) if fold_results[N]["ndcg"] else 0

    final_coverage = len(total_recommended_api_set[N]) / len(all_api_ids) if all_api_ids else 0

    print(f"N={N} -> Precision: {avg_prec:.4f}, Recall: {avg_recall:.4f}, "
          f"MAP: {avg_map:.4f}, NDCG: {avg_ndcg:.4f}, Cov: {final_coverage:.4f}")
