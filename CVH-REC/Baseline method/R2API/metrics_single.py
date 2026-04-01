import json
import math
from utility.parser import parse_args
args = parse_args()


if __name__ == '__main__':
    Ns = [3, 5, 10, 20]  # 支持多个 N 值

    # 加载 API ID -> 名称 映射
    with open("Original Dataset/apis.json", "r", encoding="utf-8") as f:
        api_data = json.load(f)
        api_id_to_title = {item["id"]: item["url"] for item in api_data}
        all_api_ids = set(api_id_to_title.keys())

    # 加载 ground truth
    fold = "fold1"
    ground_truth = {}
    with open(f"dataset/{fold}/RS.csv", "r") as f:
        for line in f:
            parts = line.strip().split()
            mashup_id = int(parts[0])
            apis = list(map(int, parts[1:]))
            ground_truth[mashup_id] = apis

    recommendations = {}
    with open(
            f"output/R2API_{fold}.json",
            "r") as f:
        for line in f:
            data = json.loads(line)
            mashup_id = data["mashup_id"]
            recommend_api = data["recommend_api"]
            recommendations[mashup_id] = recommend_api

    precision_avg_list = []
    recall_avg_list = []
    map_avg_list = []
    ndcg_avg_list = []
    coverage_avg_list = []
    # 针对每个 N 计算指标
    print(f"The performance of {args.dataset} are as follows:")
    for N in Ns:
        precision_list = []
        recall_list = []
        map_list = []
        ndcg_list = []
        recommended_api_set = set()  # 用于计算 Coverage

        for mashup_id, true_apis in ground_truth.items():
            recommended_apis = recommendations.get(mashup_id, [])[:N]
            if not true_apis:
                continue

            true_set = set(true_apis)
            hit_count = len(set(recommended_apis) & true_set)

            # 加入当前推荐结果用于计算 Coverage
            recommended_api_set.update(recommended_apis)

            # Precision@N
            precision = hit_count / N
            precision_list.append(precision)

            # Recall@N
            recall = hit_count / len(true_apis)
            recall_list.append(recall)

            # MAP@N
            cor_list = [1.0 if recommended_apis[i] in true_set else 0.0
                        for i in range(len(recommended_apis))]
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
        coverage = len(recommended_api_set) / len(all_api_ids) if all_api_ids else 0

        print(f"N={N} -> Precision: {average_precision:.4f}, Recall: {average_recall:.4f}, "
              f"MAP: {average_map:.4f}, NDCG: {average_ndcg:.4f}, Cov: {coverage:.4f}")

        # 保存用于后续平均
        precision_avg_list.append(average_precision)
        recall_avg_list.append(average_recall)
        map_avg_list.append(average_map)
        ndcg_avg_list.append(average_ndcg)
        coverage_avg_list.append(coverage)


