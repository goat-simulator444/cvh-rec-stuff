import json
import random

random_seed=1
random.seed(random_seed)

# def get_indices():
#     with open('./data/mashup_name.json', 'r') as file:
#         dataset = json.load(file)
#     split_num = int(len(dataset) / 10)
#     test_idx = dataset[:split_num]
#     train_idx = dataset[split_num:]
#     print("len(train_idx), len(test_idx)----------------------", len(train_idx), len(test_idx))
#
#
#
#     train_apis = set()
#     oov_api = set()
#     print('oov {}'.format(len(oov_api)))
#     return train_idx, test_idx, oov_api

def get_indices():
    fold = 9
    n_splits = 10
    assert 0 <= fold < n_splits, f"fold 必须在 0~{n_splits - 1} 之间"

    with open('./data/mashup_name.json', 'r') as file:
        dataset = json.load(file)

    total_size = len(dataset)
    split_size = total_size // n_splits

    # 获取测试集索引范围
    test_start = fold * split_size
    test_end = (fold + 1) * split_size if fold < n_splits - 1 else total_size

    test_idx = dataset[test_start:test_end]
    train_idx = dataset[:test_start] + dataset[test_end:]

    print(f"Fold {fold}: len(train_idx) = {len(train_idx)}, len(test_idx) = {len(test_idx)}")

    train_apis = set()
    oov_api = set()
    print('oov {}'.format(len(oov_api)))

    return train_idx, test_idx, oov_api

