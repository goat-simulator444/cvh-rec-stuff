import json
import math
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from math import ceil
from scipy.stats import ttest_rel, wilcoxon, normaltest
from statsmodels.stats.multitest import multipletests

Ns = [3, 5, 10, 20]
folds = ["fold1", "fold2", "fold3", "fold4", "fold5", "fold6", "fold7", "fold8", "fold9", "fold10"]


def compute_method_metrics_per_mashup(method, folds):
    metrics_dict = {m: {} for m in ["precision", "recall", "map", "ndcg"]}

    for fold in folds:
        # load ground truth
        ground_truth = {}
        with open(f"dataset/{fold}/RS.csv", "r") as f:
            for line in f:
                parts = line.strip().split()
                mashup_id = int(parts[0])
                apis = list(map(int, parts[1:]))
                ground_truth[mashup_id] = apis

        # load recommendations
        file_name = f"{method}_{fold}"
        recommendations = {}
        with open(f"output/{file_name}.json", "r") as f:
            for line in f:
                data = json.loads(line)
                mashup_id = data["mashup_id"]
                recommendations[mashup_id] = data["recommend_api"]

        # 每个 mashup 的指标
        precision_list, recall_list, map_list, ndcg_list = [], [], [], []

        for mashup_id, true_apis in ground_truth.items():
            if not true_apis:
                continue
            true_set = set(true_apis)

            prec_vec, rec_vec, map_vec, ndcg_vec = [], [], [], []

            for N in Ns:
                recommended = recommendations.get(mashup_id, [])[:N]

                hit_count = len(set(recommended) & true_set)
                # Precision
                prec_vec.append(hit_count / N)
                # Recall
                rec_vec.append(hit_count / len(true_set))
                # MAP
                cor_list = [1.0 if api in true_set else 0.0 for api in recommended]
                sum_cor = sum(cor_list)
                if sum_cor == 0:
                    map_score = 0.0
                else:
                    map_score = sum(
                        sum(cor_list[:i + 1]) / (i + 1) * cor_list[i] for i in range(len(cor_list))) / sum_cor
                map_vec.append(map_score)
                # NDCG
                dcg = sum(1 / math.log2(i + 2) if i < len(recommended) and recommended[i] in true_set else 0
                          for i in range(N))
                idcg = sum(1 / math.log2(i + 2) for i in range(min(len(true_set), N)))
                ndcg_vec.append(dcg / idcg if idcg != 0 else 0)

            precision_list.append(np.array(prec_vec))
            recall_list.append(np.array(rec_vec))
            map_list.append(np.array(map_vec))
            ndcg_list.append(np.array(ndcg_vec))

        metrics_dict["precision"][fold] = precision_list
        metrics_dict["recall"][fold] = recall_list
        metrics_dict["map"][fold] = map_list
        metrics_dict["ndcg"][fold] = ndcg_list

    return metrics_dict


def compare_methods_fold_mean(method_a, method_b, folds):
    mA = compute_method_metrics_per_mashup(method_a, folds)
    mB = compute_method_metrics_per_mashup(method_b, folds)

    metrics = ["precision", "recall", "map", "ndcg"]

    for idx, N in enumerate(Ns):
        for metric in metrics:
            a_fold_means = np.array([np.mean([m[idx] for m in mA[metric][fold]]) for fold in folds])
            b_fold_means = np.array([np.mean([m[idx] for m in mB[metric][fold]]) for fold in folds])

            from scipy.stats import normaltest

            diff = a_fold_means - b_fold_means

            # stat, pval = ttest_rel(a_fold_means, b_fold_means)
            stat, pval = wilcoxon(a_fold_means, b_fold_means, zero_method="pratt", correction=False)

            def cohen(a, b):
                a = np.array(a)
                b = np.array(b)
                sd_a = np.std(a, ddof=1)
                sd_b = np.std(b, ddof=1)
                sd_pooled = np.sqrt((sd_a ** 2 + sd_b ** 2) / 2)
                mean_diff = np.mean(np.array(a) - np.array(b))
                n = len(a)
                J = 1 - 3 / (4 * n - 1)
                hedges_g = (mean_diff / sd_pooled) * J
                return hedges_g

            d_val = cohen(a_fold_means, b_fold_means)
            print(
                f"{metric:<7} {N:<3} p-value: {pval:.2e} Cohen's d value: {d_val:7.4f}"
            )


import argparse
def my_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method1", type=str, required=True, help="Name of the first method (e.g., CVH-REC)")
    parser.add_argument("--method2", type=str, required=True, help="Name of the second method (e.g., R2API)")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    my_args = my_parse_args()
    method1 = my_args.method1
    method2 = my_args.method2

    compare_methods_fold_mean(method1, method2, folds)