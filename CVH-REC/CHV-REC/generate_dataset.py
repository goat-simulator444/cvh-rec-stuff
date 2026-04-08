import argparse
import json
import os
from math import ceil

import h5py
import numpy as np
from sentence_transformers import SentenceTransformer

from utility.text_model_utils import embedding_dir, resolve_text_model_name, text_model_tag


def parse_args():
    parser = argparse.ArgumentParser(description="Generate CVH-REC dataset and text embeddings.")
    parser.add_argument(
        "--text_model",
        default="mpnet",
        help="Text model alias or HF name (e.g., mpnet, bge-base, bge-large, e5-base, e5-large, gte-base, gte-large).",
    )
    parser.add_argument(
        "--embedding_base_dir",
        default="data",
        help="Base directory used to store encoder-specific h5 embedding files.",
    )
    return parser.parse_args()


def save_vectors_h5(model, filename, id_to_text):
    with h5py.File(filename, "w") as f:
        for id_, text in id_to_text.items():
            vec = model.encode(text)
            f.create_dataset(str(id_), data=vec.astype(np.float32))


def main():
    args = parse_args()

    # Load source data.
    with open("Original Dataset/mashup_shuffled.json", "r", encoding="utf-8") as f:
        mashups = json.load(f)

    with open("Original Dataset/apis.json", "r", encoding="utf-8") as f:
        apis = json.load(f)

    # Build tag dictionaries.
    api_id_to_tags = {}
    all_api_tags = set()
    for api in apis:
        api_id = api["id"]
        tags_str = api["details"].get("tags", "")
        tags = [t.strip() for t in tags_str.split(",") if t.strip()]
        api_id_to_tags[api_id] = tags
        all_api_tags.update(tags)

    mashup_id_to_tags = {}
    all_mashup_tags = set()
    for mashup in mashups:
        mashup_id = mashup["id"]
        cats = [c.strip() for c in mashup["categories"].split(",") if c.strip()]
        mashup_id_to_tags[mashup_id] = cats
        all_mashup_tags.update(cats)

    all_tags = all_api_tags.union(all_mashup_tags)
    tag2id = {tag: i for i, tag in enumerate(sorted(all_tags))}

    model_name = resolve_text_model_name(args.text_model)
    model_tag = text_model_tag(args.text_model)
    print(f"Using text model: {model_name} (tag={model_tag})")

    # Keep embeddings isolated per text model to avoid dimension collisions.
    model_embedding_dir = embedding_dir(args.embedding_base_dir, args.text_model)
    os.makedirs(model_embedding_dir, exist_ok=True)

    model = SentenceTransformer(model_name)

    save_vectors_h5(model, os.path.join(model_embedding_dir, "tag_vectors.h5"), {tid: tag for tag, tid in tag2id.items()})

    api_id_to_desc = {api["id"]: api["details"].get("description", "") for api in apis}
    save_vectors_h5(model, os.path.join(model_embedding_dir, "API_vectors.h5"), api_id_to_desc)

    mashup_id_to_desc = {mashup["id"]: mashup.get("description", "") for mashup in mashups}
    save_vectors_h5(model, os.path.join(model_embedding_dir, "vectors.h5"), mashup_id_to_desc)

    n_total = len(mashups)
    split_idx = round(n_total * 0.9)
    cv_mashups = mashups[:split_idx]
    folds = 10
    fold_size = ceil(len(cv_mashups) / folds)

    for i in range(folds):
        fold_dir = f"dataset/fold{i + 1}"
        os.makedirs(fold_dir, exist_ok=True)

        start = i * fold_size
        end = min((i + 1) * fold_size, len(cv_mashups))

        test_set = cv_mashups[start:end]
        train_set = cv_mashups[:start] + cv_mashups[end:]

        with open(os.path.join(fold_dir, "TE.csv"), "w", encoding="utf-8") as f:
            for mashup in train_set:
                line = [str(mashup["id"])] + [str(api_id) for api_id in mashup.get("api_info", [])]
                f.write(" ".join(line) + "\n")

        with open(os.path.join(fold_dir, "RS.csv"), "w", encoding="utf-8") as f:
            for mashup in test_set:
                line = [str(mashup["id"])] + [str(api_id) for api_id in mashup.get("api_info", [])]
                f.write(" ".join(line) + "\n")

        with open(os.path.join(fold_dir, "VA.csv"), "w", encoding="utf-8") as f:
            for mashup in mashups[split_idx:]:
                line = [str(mashup["id"])] + [str(api_id) for api_id in mashup.get("api_info", [])]
                f.write(" ".join(line) + "\n")

        with open(os.path.join(fold_dir, "api_tags.csv"), "w", encoding="utf-8") as f:
            for api_id, tags in api_id_to_tags.items():
                tag_ids = [str(tag2id[t]) for t in tags if t in tag2id]
                f.write(f"{api_id} " + " ".join(tag_ids) + "\n")

        with open(os.path.join(fold_dir, "mashup_tags.csv"), "w", encoding="utf-8") as f:
            for mashup in train_set:
                tags = mashup_id_to_tags.get(mashup["id"], [])
                tag_ids = [str(tag2id[t]) for t in tags if t in tag2id]
                f.write(f"{mashup['id']} " + " ".join(tag_ids) + "\n")


if __name__ == "__main__":
    main()

