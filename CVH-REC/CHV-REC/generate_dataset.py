import argparse
import json
import os
from math import ceil

import h5py
import numpy as np
from sentence_transformers import SentenceTransformer

from utility.text_model_utils import embedding_dir, resolve_text_model_name, text_model_tag


def parse_args():
    parser = argparse.ArgumentParser(description="Generate CVH-REC dataset and text vectors.")
    parser.add_argument(
        "--text_model",
        default="mpnet",
        help="Model alias/name. Supported aliases: mpnet, bge-base, bge-large, e5-base (or e5bse), e5-large, gte-base, gte-large, all.",
    )
    parser.add_argument(
        "--embedding_base_dir",
        default="data",
        help="Base directory where per-model embedding folders are created.",
    )
    return parser.parse_args()


def load_source_data():
    with open("Original Dataset/mashup_shuffled.json", "r", encoding="utf-8") as f:
        mashups = json.load(f)
    with open("Original Dataset/apis.json", "r", encoding="utf-8") as f:
        apis = json.load(f)
    return mashups, apis


def build_tag_maps(mashups, apis):
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
        categories = [c.strip() for c in mashup["categories"].split(",") if c.strip()]
        mashup_id_to_tags[mashup_id] = categories
        all_mashup_tags.update(categories)

    all_tags = all_api_tags.union(all_mashup_tags)
    tag2id = {tag: i for i, tag in enumerate(sorted(all_tags))}
    return api_id_to_tags, mashup_id_to_tags, tag2id


def save_vectors_h5(model, filename, id_to_text):
    with h5py.File(filename, "w") as f:
        for id_, text in id_to_text.items():
            vec = model.encode(text)
            f.create_dataset(str(id_), data=vec.astype(np.float32))


def generate_embeddings(mashups, apis, tag2id, text_model, embedding_base_dir):
    model_name = resolve_text_model_name(text_model)
    tag = text_model_tag(text_model)
    out_dir = embedding_dir(embedding_base_dir, text_model)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Generating vectors with model={model_name} (tag={tag}) -> {out_dir}")
    model = SentenceTransformer(model_name)

    save_vectors_h5(
        model,
        os.path.join(out_dir, "tag_vectors.h5"),
        {tid: tag_text for tag_text, tid in tag2id.items()},
    )

    api_id_to_desc = {api["id"]: api["details"]["description"] for api in apis}
    save_vectors_h5(model, os.path.join(out_dir, "API_vectors.h5"), api_id_to_desc)

    mashup_id_to_desc = {m["id"]: m["description"] for m in mashups}
    save_vectors_h5(model, os.path.join(out_dir, "vectors.h5"), mashup_id_to_desc)


def generate_folds(mashups, api_id_to_tags, mashup_id_to_tags, tag2id):
    n_mashups = len(mashups)
    train_valid_boundary = round(n_mashups * 0.9)
    cv_mashups = mashups[:train_valid_boundary]

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
            for mashup in mashups[end:]:
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


def resolve_models_to_generate(text_model):
    if str(text_model).strip().lower() == "all":
        return ["bge-base", "e5-base", "gte-base", "bge-large", "e5-large", "gte-large"]
    return [text_model]


def main():
    args = parse_args()

    mashups, apis = load_source_data()
    api_id_to_tags, mashup_id_to_tags, tag2id = build_tag_maps(mashups, apis)

    for model_name in resolve_models_to_generate(args.text_model):
        generate_embeddings(mashups, apis, tag2id, model_name, args.embedding_base_dir)

    # Interaction/tag split files are model-agnostic and generated once.
    generate_folds(mashups, api_id_to_tags, mashup_id_to_tags, tag2id)


if __name__ == "__main__":
    main()

