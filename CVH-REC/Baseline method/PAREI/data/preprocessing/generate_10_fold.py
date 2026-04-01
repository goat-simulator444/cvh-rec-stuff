def split_mashup_ids():
    total_ids = list(range(7163))  # 0 到 7162
    num_folds = 10
    fold_size = len(total_ids) // num_folds
    remainder = len(total_ids) % num_folds

    folds = []
    start = 0
    for i in range(num_folds):
        end = start + fold_size + (1 if i < remainder else 0)  # 前 remainder 折多一个
        folds.append(total_ids[start:end])
        start = end

    return folds

# 输出查看
folds = split_mashup_ids()
for i, fold in enumerate(folds):
    print(f"{fold}")

