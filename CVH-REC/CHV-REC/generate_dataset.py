import argparse
import json
import os
from datetime import datetime
from math import ceil

import h5py
import numpy as np

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
    parser.add_argument(
        "--write_foldwise_timestamp_maps",
        type=int,
        default=1,
        help="Whether to write fold-wise timestamp map files for train/test/val splits (1=yes, 0=no).",
    )
    parser.add_argument(
        "--fold_timestamp_maps_subdir",
        default="timestamp_maps",
        help="Subdirectory name inside each dataset/foldX directory to store fold-wise timestamp map files.",
    )
    parser.add_argument(
        "--skip_embedding_generation",
        type=int,
        default=0,
        help="Whether to skip embedding generation and only rebuild dataset split files (1=yes, 0=no).",
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


def build_temporal_records(mashup_rows, timestamp_map, timestamp_fallback):
    interaction_order = 0
    edge_count = 0
    map_edges = 0
    fallback_edges = 0
    records = []

    for mashup in mashup_rows:
        mashup_id = int(mashup["id"])
        for api_id in mashup.get("api_info", []):
            timestamp, source = _resolve_timestamp(
                mashup_id,
                interaction_order,
                timestamp_map,
                timestamp_fallback,
            )
            records.append(
                {
                    "mashup_id": mashup_id,
                    "api_id": int(api_id),
                    "timestamp": float(timestamp),
                    "source": source,
                }
            )

            edge_count += 1
            interaction_order += 1
            if source == "map":
                map_edges += 1
            else:
                fallback_edges += 1

    stats = {
        "edges": edge_count,
        "map_edges": map_edges,
        "fallback_edges": fallback_edges,
    }
    return records, stats


def write_temporal_interactions(file_path, temporal_records):
    with open(file_path, "w", encoding="utf-8") as f:
        for record in temporal_records:
            f.write(f"{record['mashup_id']} {record['api_id']} {record['timestamp']}\n")


def _build_fold_timestamp_map(mashup_rows, timestamp_map):
    fold_map = {}
    missing = []

    for mashup in mashup_rows:
        mashup_id = int(mashup["id"])
        if mashup_id in timestamp_map:
            fold_map[str(mashup_id)] = float(timestamp_map[mashup_id])
        else:
            missing.append(mashup_id)

    return fold_map, missing


def _write_json(file_path, obj):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def _write_foldwise_timestamp_maps(
    fold_dir,
    timestamp_maps_subdir,
    train_set,
    test_set,
    val_set,
    timestamp_map,
    timestamp_map_file,
    timestamp_fallback,
    temporal_stats,
):
    maps_dir = os.path.join(fold_dir, timestamp_maps_subdir)
    os.makedirs(maps_dir, exist_ok=True)

    train_map, train_missing = _build_fold_timestamp_map(train_set, timestamp_map)
    test_map, test_missing = _build_fold_timestamp_map(test_set, timestamp_map)
    val_map, val_missing = _build_fold_timestamp_map(val_set, timestamp_map)

    all_mashup_ids = sorted({int(m["id"]) for m in (train_set + test_set + val_set)})
    all_map = {}
    all_map.update(train_map)
    all_map.update(test_map)
    all_map.update(val_map)
    all_missing = [mid for mid in all_mashup_ids if str(mid) not in all_map]

    def _coverage(mapped_count, total_count):
        if total_count == 0:
            return 0.0
        return float(mapped_count) / float(total_count)

    report = {
        "timestamp_map_file": timestamp_map_file if timestamp_map_file else None,
        "timestamp_map_entries_loaded": len(timestamp_map),
        "timestamp_fallback": timestamp_fallback,
        "splits": {
            "train": {
                "total_mashups": len(train_set),
                "mapped_mashups": len(train_map),
                "missing_mashups": len(train_missing),
                "coverage": _coverage(len(train_map), len(train_set)),
                "missing_mashup_ids": train_missing,
            },
            "test": {
                "total_mashups": len(test_set),
                "mapped_mashups": len(test_map),
                "missing_mashups": len(test_missing),
                "coverage": _coverage(len(test_map), len(test_set)),
                "missing_mashup_ids": test_missing,
            },
            "val": {
                "total_mashups": len(val_set),
                "mapped_mashups": len(val_map),
                "missing_mashups": len(val_missing),
                "coverage": _coverage(len(val_map), len(val_set)),
                "missing_mashup_ids": val_missing,
            },
            "all": {
                "total_mashups": len(all_mashup_ids),
                "mapped_mashups": len(all_map),
                "missing_mashups": len(all_missing),
                "coverage": _coverage(len(all_map), len(all_mashup_ids)),
                "missing_mashup_ids": all_missing,
            },
        },
        "temporal_edges": temporal_stats,
    }

    _write_json(os.path.join(maps_dir, "train.json"), train_map)
    _write_json(os.path.join(maps_dir, "test.json"), test_map)
    _write_json(os.path.join(maps_dir, "val.json"), val_map)
    _write_json(os.path.join(maps_dir, "all.json"), all_map)
    _write_json(os.path.join(maps_dir, "coverage_report.json"), report)


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

    if args.skip_embedding_generation != 1:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            raise RuntimeError(
                "Failed to import sentence_transformers. "
                "Install a compatible environment or run with --skip_embedding_generation 1."
            ) from exc

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
    else:
        print("Skipping embedding generation (--skip_embedding_generation=1).")

    n_total = len(mashups)
    split_idx = round(n_total * 0.9)
    cv_mashups = mashups[:split_idx]
    val_set = mashups[split_idx:]
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
        write_static_interactions(os.path.join(fold_dir, "VA.csv"), val_set)

        te_stats = {"edges": 0, "map_edges": 0, "fallback_edges": 0}
        rs_stats = {"edges": 0, "map_edges": 0, "fallback_edges": 0}
        va_stats = {"edges": 0, "map_edges": 0, "fallback_edges": 0}

        if args.write_temporal_files == 1:
            te_records, te_stats = build_temporal_records(
                train_set,
                timestamp_map,
                args.timestamp_fallback,
            )
            rs_records, rs_stats = build_temporal_records(
                test_set,
                timestamp_map,
                args.timestamp_fallback,
            )
            va_records, va_stats = build_temporal_records(
                val_set,
                timestamp_map,
                args.timestamp_fallback,
            )

            write_temporal_interactions(os.path.join(fold_dir, "TE_temporal.csv"), te_records)
            write_temporal_interactions(os.path.join(fold_dir, "RS_temporal.csv"), rs_records)
            write_temporal_interactions(os.path.join(fold_dir, "VA_temporal.csv"), va_records)

            print(
                f"[fold{i + 1}] Temporal files generated | "
                f"TE edges={te_stats['edges']} (map={te_stats['map_edges']}, fallback={te_stats['fallback_edges']}) | "
                f"RS edges={rs_stats['edges']} (map={rs_stats['map_edges']}, fallback={rs_stats['fallback_edges']}) | "
                f"VA edges={va_stats['edges']} (map={va_stats['map_edges']}, fallback={va_stats['fallback_edges']})"
            )

        if args.write_foldwise_timestamp_maps == 1:
            _write_foldwise_timestamp_maps(
                fold_dir=fold_dir,
                timestamp_maps_subdir=args.fold_timestamp_maps_subdir,
                train_set=train_set,
                test_set=test_set,
                val_set=val_set,
                timestamp_map=timestamp_map,
                timestamp_map_file=args.timestamp_map_file,
                timestamp_fallback=args.timestamp_fallback,
                temporal_stats={
                    "TE": te_stats,
                    "RS": rs_stats,
                    "VA": va_stats,
                },
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

