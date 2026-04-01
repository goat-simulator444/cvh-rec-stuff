import json
import math
import numpy as np
import os
from utility.parser import parse_args

args = parse_args()

if __name__ == '__main__':
    Ns = [3, 5, 10, 20]
    method = args.method

    # === 用于存储每一折结果 ===
    precision_folds = []
    recall_folds = []
    map_folds = []
    ndcg_folds = []

    for i in range(1, 11):
        fold = "fold" + str(i)
        file_name = f'{method}_{fold}'
        file_path = f"{args.output}/{file_name}.json"
        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist. Please run the corresponding command to obtain the recommendation results.")
            continue
        # === 加载 ground truth ===
        ground_truth = {}
        with open(f"dataset/{fold}/RS.csv", "r") as f:
            for line in f:
                parts = line.strip().split()
                mashup_id = int(parts[0])
                apis = list(map(int, parts[1:]))
                ground_truth[mashup_id] = apis

        recommendations = {}
        with open(f"{args.output}/{file_name}.json", "r") as f:
            for line in f:
                data = json.loads(line)
                mashup_id = data["mashup_id"]
                recommend_api = data["recommend_api"]
                recommendations[mashup_id] = recommend_api
        precision_all, recall_all, map_all, ndcg_all = [], [], [], []

        for N in Ns:
            precision_list, recall_list, map_list, ndcg_list = [], [], [], []

            for mashup_id, true_apis in ground_truth.items():
                recommended_apis = recommendations.get(mashup_id, [])[:N]

                true_set = set(true_apis)
                hit_count = len(set(recommended_apis) & true_set)

                # === Precision@N ===
                precision_list.append(hit_count / N)
                # === Recall@N ===
                recall_list.append(hit_count / len(true_apis))

                # === MAP@N ===
                cor_list = [1.0 if api in true_set else 0.0 for api in recommended_apis]
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

                # === NDCG@N ===
                dcg = sum(
                    1 / math.log2(i + 2)
                    if i < len(recommended_apis) and recommended_apis[i] in true_set
                    else 0
                    for i in range(N)
                )
                idcg = sum(1 / math.log2(i + 2) for i in range(min(len(true_apis), N)))
                ndcg_list.append(dcg / idcg if idcg != 0 else 0)

            # === 每折的平均指标 ===
            precision_all.append(np.mean(precision_list) if precision_list else 0)
            recall_all.append(np.mean(recall_list) if recall_list else 0)
            map_all.append(np.mean(map_list) if map_list else 0)
            ndcg_all.append(np.mean(ndcg_list) if ndcg_list else 0)

        precision_folds.append(precision_all)
        recall_folds.append(recall_all)
        map_folds.append(map_all)
        ndcg_folds.append(ndcg_all)
    if len(precision_folds) == 0:
        print("The is no recommendation result! Please run the corresponding command.")
        exit()
    # === 平均与标准差 ===
    precision_mean = np.mean(precision_folds, axis=0)
    precision_std = np.std(precision_folds, axis=0)
    recall_mean = np.mean(recall_folds, axis=0)
    recall_std = np.std(recall_folds, axis=0)
    map_mean = np.mean(map_folds, axis=0)
    map_std = np.std(map_folds, axis=0)
    ndcg_mean = np.mean(ndcg_folds, axis=0)
    ndcg_std = np.std(ndcg_folds, axis=0)

    # === 输出结果 ===
    # def format_metric_line(name, mean, std):
    #     formatted = " ".join(f"{m:.4f}±{s:.4f}" for m, s in zip(mean, std))
    #     return f"{name:<10}: {formatted}"
    #
    # print("N values   :", " ".join(f"{n:<10}" for n in Ns))
    # print(format_metric_line("Precision", precision_mean, precision_std))
    # print(format_metric_line("Recall", recall_mean, recall_std))
    # print(format_metric_line("MAP", map_mean, map_std))
    # print(format_metric_line("NDCG", ndcg_mean, ndcg_std))

    def format_metric_line(name, mean, std):
        # 每个数值块固定宽度 13，包括“±”符号
        formatted = " ".join(f"{m:7.4f}±{s:<6.4f}" for m, s in zip(mean, std))
        return f"{name:<10}: {formatted}"


    # 打印标题行
    print(f"{'N values':<10}: " + " ".join(f"{n:<16}" for n in Ns))
    print(format_metric_line("Precision", precision_mean, precision_std))
    print(format_metric_line("Recall", recall_mean, recall_std))
    print(format_metric_line("MAP", map_mean, map_std))
    print(format_metric_line("NDCG", ndcg_mean, ndcg_std))

