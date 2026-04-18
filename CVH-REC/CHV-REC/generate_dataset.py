import argparse
import json
import os
from datetime import datetime
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
    parser.add_argument(
        "--write_temporal_files",
        type=int,
        default=1,
        help="Whether to write temporal triplet files (1=yes, 0=no).",
    )
    parser.add_argument(
        "--timestamp_map_file",
        default="",
        help="Optional JSON file mapping mashup_id to timestamp for explicit temporal data.",
    )
    parser.add_argument(
        "--timestamp_fallback",
        choices=["line_order", "mashup_id"],
        default="line_order",
        help="Fallback timestamp strategy when timestamp_map_file is missing or incomplete.",
    )
    return parser.parse_args()


def save_vectors_h5(model, filename, id_to_text):
    with h5py.File(filename, "w") as f:
        for id_, text in id_to_text.items():
            vec = model.encode(text)
            f.create_dataset(str(id_), data=vec.astype(np.float32))


def _parse_timestamp_value(value):
    if value is None:
        return None

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError:
            pass
        # Accept ISO-like strings (e.g., 2024-01-02 or 2024-01-02T12:00:00Z).
        try:
            dt = datetime.fromisoformat(stripped.replace("Z", "+00:00"))
            return float(dt.timestamp())
        except ValueError:
            return None

    return None


def load_timestamp_map(timestamp_map_file):
    if not timestamp_map_file:
        return {}

    with open(timestamp_map_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    timestamp_map = {}

    if isinstance(data, dict):
        for mashup_id, ts in data.items():
            try:
                mashup_id_int = int(mashup_id)
            except (TypeError, ValueError):
                continue

            ts_parsed = _parse_timestamp_value(ts)
            if ts_parsed is not None:
                timestamp_map[mashup_id_int] = ts_parsed

    elif isinstance(data, list):
        for row in data:
            if not isinstance(row, dict):
                continue

            mashup_id = row.get("mashup_id", row.get("id"))
            if mashup_id is None:
                continue

            try:
                mashup_id_int = int(mashup_id)
            except (TypeError, ValueError):
                continue

            ts_raw = row.get("timestamp", row.get("time", row.get("date")))
            ts_parsed = _parse_timestamp_value(ts_raw)
            if ts_parsed is not None:
                timestamp_map[mashup_id_int] = ts_parsed

    return timestamp_map


def write_static_interactions(file_path, mashup_rows):
    with open(file_path, "w", encoding="utf-8") as f:
        for mashup in mashup_rows:
            line = [str(mashup["id"])] + [str(api_id) for api_id in mashup.get("api_info", [])]
            f.write(" ".join(line) + "\n")


def _resolve_timestamp(mashup_id, interaction_order, timestamp_map, timestamp_fallback):
    if mashup_id in timestamp_map:
        return float(timestamp_map[mashup_id]), "map"

    if timestamp_fallback == "line_order":
        return float(interaction_order), "fallback"
    if timestamp_fallback == "mashup_id":
        return float(mashup_id), "fallback"

    raise ValueError(f"Unsupported timestamp_fallback: {timestamp_fallback}")


def write_temporal_interactions(file_path, mashup_rows, timestamp_map, timestamp_fallback):
    interaction_order = 0
    edge_count = 0
    map_edges = 0
    fallback_edges = 0

    with open(file_path, "w", encoding="utf-8") as f:
        for mashup in mashup_rows:
            mashup_id = int(mashup["id"])
            for api_id in mashup.get("api_info", []):
                timestamp, source = _resolve_timestamp(
                    mashup_id,
                    interaction_order,
                    timestamp_map,
                    timestamp_fallback,
                )
                f.write(f"{mashup_id} {int(api_id)} {timestamp}\n")

                edge_count += 1
                interaction_order += 1
                if source == "map":
                    map_edges += 1
                else:
                    fallback_edges += 1

    return {
        "edges": edge_count,
        "map_edges": map_edges,
        "fallback_edges": fallback_edges,
    }


def main():
    args = parse_args()

    timestamp_map = load_timestamp_map(args.timestamp_map_file)
    if args.timestamp_map_file:
        print(
            f"Timestamp method: explicit_map({args.timestamp_map_file}) + fallback:{args.timestamp_fallback}; "
            f"loaded={len(timestamp_map)} mashup timestamps"
        )
    else:
        print(f"Timestamp method: generated:{args.timestamp_fallback}")

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

        write_static_interactions(os.path.join(fold_dir, "TE.csv"), train_set)
        write_static_interactions(os.path.join(fold_dir, "RS.csv"), test_set)
        write_static_interactions(os.path.join(fold_dir, "VA.csv"), mashups[split_idx:])

        if args.write_temporal_files == 1:
            te_stats = write_temporal_interactions(
                os.path.join(fold_dir, "TE_temporal.csv"),
                train_set,
                timestamp_map,
                args.timestamp_fallback,
            )
            rs_stats = write_temporal_interactions(
                os.path.join(fold_dir, "RS_temporal.csv"),
                test_set,
                timestamp_map,
                args.timestamp_fallback,
            )
            va_stats = write_temporal_interactions(
                os.path.join(fold_dir, "VA_temporal.csv"),
                mashups[split_idx:],
                timestamp_map,
                args.timestamp_fallback,
            )
            print(
                f"[fold{i + 1}] Temporal files generated | "
                f"TE edges={te_stats['edges']} (map={te_stats['map_edges']}, fallback={te_stats['fallback_edges']}) | "
                f"RS edges={rs_stats['edges']} (map={rs_stats['map_edges']}, fallback={rs_stats['fallback_edges']}) | "
                f"VA edges={va_stats['edges']} (map={va_stats['map_edges']}, fallback={va_stats['fallback_edges']})"
            )

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

