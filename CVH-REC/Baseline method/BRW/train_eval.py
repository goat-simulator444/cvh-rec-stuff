import numpy as np
from sklearn.model_selection import KFold
from data_loader import load_data, build_graph
from graph_model import generate_embeddings

def recommend_top_n(model, mashup_ids, api_ids, N=20):
    scores = {}
    for m in mashup_ids:
        if m not in model.wv: continue
        m_vec = model.wv[m]
        scores[m] = sorted([(a, np.dot(m_vec, model.wv[a])) for a in api_ids if a in model.wv], key=lambda x: -x[1])[:N]
    return scores


def evaluate_recall(recommendations, ground_truth, ks=(3, 5, 10, 20)):
    recalls = {k: [] for k in ks}
    for m_id, recs in recommendations.items():
        pred_apis = [a for a, _ in recs]
        true_apis = ground_truth.get(m_id, set())
        if not true_apis:
            continue
        for k in ks:
            hits = len(set(pred_apis[:k]) & true_apis)
            recalls[k].append(hits / len(true_apis))
    return {f"Recall@{k}": np.mean(recalls[k]) if recalls[k] else 0.0 for k in ks}

import json

def run_cross_validation(mashup_path="data/shuffle_mashup_details.json", api_path="data/api_id_mapping.json", folds=10, topN=20):
    mashups, apis = load_data(mashup_path, api_path)
    mashup_ids = [m['id'] for m in mashups]
    mashup_dict = {m['id']: m for m in mashups}
    kf = KFold(n_splits=folds)

    for fold, (train_idx, test_idx) in enumerate(kf.split(mashup_ids)):
        print(f"\nFold {fold+1}/{folds}")

        train_ids = {mashup_ids[i] for i in train_idx}
        test_ids = {mashup_ids[i] for i in test_idx}

        G, mashup_nodes, api_nodes = build_graph(mashups, apis, test_ids=test_ids)

        model = generate_embeddings(G, dimensions=128, walk_length=20, num_walks=80, phi=0.4, omega=0.6)

        test_mashup_nodes = [f"m_{mid}" for mid in test_ids]
        recs = recommend_top_n(model, test_mashup_nodes, api_nodes, N=topN)


        # 构建 ground truth
        gt = {}
        for mid in test_ids:
            mkey = f"m_{mid}"
            if mkey in G:
                true_apis = {nbr for nbr in G.neighbors(mkey)
                             if G.edges[mkey, nbr]['type'] == 'call'}
                gt[mkey] = true_apis

        # 计算 recall
        # recall_result = evaluate_recall(recs, gt)
        # print("Recall:", recall_result)

        # 保存推荐结果为 JSONL
        output_path = f"test_random_walk_{fold+1}.jsonl"
        with open(output_path, 'w', encoding='utf-8') as f:
            for mkey, ranked_apis in recs.items():
                mid = int(mkey.split('_')[1])
                rec_api_ids = [int(a.split('_')[1]) for a, _ in ranked_apis]
                removed_api_ids = [int(nbr.split('_')[1]) for nbr in gt.get(mkey, [])]
                result = {
                    "mashup_id": mid,
                    "remove_apis": removed_api_ids,
                    "recommend_api": rec_api_ids
                }
                f.write(json.dumps(result) + '\n')


if __name__ == "__main__":
    run_cross_validation("data/shuffle_mashup_details.json", "data/api_id_mapping.json")

