import argparse
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from urllib.request import urlopen


DEFAULT_ACTIVE_MASHUPS_URL = (
    "https://raw.githubusercontent.com/HIT-ICES/Correted-ProgrammableWeb-dataset/main/"
    "data/raw/api_mashup/active_mashups_data.txt"
)
DEFAULT_DEAD_MASHUPS_URL = (
    "https://raw.githubusercontent.com/HIT-ICES/Correted-ProgrammableWeb-dataset/main/"
    "data/raw/api_mashup/deadpool_mashups_data.txt"
)


_WS_RE = re.compile(r"\s+")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build mashup_id->timestamp map for local CVH-REC dataset using "
            "HIT-ICES Corrected ProgrammableWeb raw mashup data."
        )
    )
    parser.add_argument(
        "--local_mashups_file",
        default="Original Dataset/mashup_shuffled.json",
        help="Path to local mashup JSON file (expects id/title/url fields).",
    )
    parser.add_argument(
        "--output_map_file",
        default="data/timestamp_map_hitices.json",
        help="Output JSON map path (mashup_id -> unix timestamp).",
    )
    parser.add_argument(
        "--output_report_file",
        default="data/timestamp_map_hitices_report.json",
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--output_unmatched_file",
        default="data/timestamp_map_hitices_unmatched.json",
        help="Output JSON unmatched mashups path.",
    )
    parser.add_argument(
        "--active_mashups_url",
        default=DEFAULT_ACTIVE_MASHUPS_URL,
        help="URL of HIT-ICES active mashups raw JSON.",
    )
    parser.add_argument(
        "--dead_mashups_url",
        default=DEFAULT_DEAD_MASHUPS_URL,
        help="URL of HIT-ICES deadpool mashups raw JSON.",
    )
    parser.add_argument(
        "--timeout_sec",
        type=int,
        default=120,
        help="HTTP timeout in seconds for downloading HIT-ICES files.",
    )
    return parser.parse_args()


def _normalize_whitespace(text):
    return _WS_RE.sub(" ", str(text or "").strip())


def normalize_title(title):
    text = _normalize_whitespace(title).lower()
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    if text.startswith("mashup:"):
        suffix = text[len("mashup:") :].strip()
        text = f"mashup: {suffix}"
    return text


def normalize_url(url):
    text = _normalize_whitespace(url).lower().strip("\n")
    if not text:
        return ""

    if text.startswith("http://"):
        text = text[len("http://") :]
    elif text.startswith("https://"):
        text = text[len("https://") :]

    if text.startswith("www."):
        text = text[len("www.") :]

    if text.endswith("/"):
        text = text[:-1]

    return text


def parse_date_to_unix(date_value):
    if date_value is None:
        return None

    if isinstance(date_value, (int, float)):
        return float(date_value)

    text = str(date_value).strip()
    if not text:
        return None

    formats = [
        "%m.%d.%Y",
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%Y/%m/%d",
        "%Y.%m.%d",
        "%d.%m.%Y",
    ]

    for fmt in formats:
        try:
            dt = datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
            if dt.year >= 2100:
                return None
            return float(dt.timestamp())
        except ValueError:
            continue

    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        if dt.year >= 2100:
            return None
        return float(dt.timestamp())
    except ValueError:
        return None


def _fetch_json(url, timeout_sec):
    with urlopen(url, timeout=timeout_sec) as response:
        payload = response.read().decode("utf-8")
    return json.loads(payload)


def _to_source_entry(row, source_split):
    title = row.get("title", "")
    url = row.get("url", "")
    date_raw = row.get("date")

    return {
        "title": title,
        "title_norm": normalize_title(title),
        "url": url,
        "url_norm": normalize_url(url),
        "date_raw": date_raw,
        "timestamp": parse_date_to_unix(date_raw),
        "source_split": source_split,
    }


def _select_candidate(candidates, local_url_norm):
    candidates_with_ts = [c for c in candidates if c["timestamp"] is not None]
    if not candidates_with_ts:
        return None, "no_timestamp"

    if local_url_norm:
        url_filtered = [c for c in candidates_with_ts if c["url_norm"] == local_url_norm]
        if len(url_filtered) == 1:
            return url_filtered[0], "url_refine"
        if len(url_filtered) > 1:
            chosen = sorted(
                url_filtered,
                key=lambda x: (x["timestamp"], x["url_norm"], x["title_norm"]),
            )[0]
            return chosen, "url_refine_ambiguous"

    if len(candidates_with_ts) == 1:
        return candidates_with_ts[0], "single"

    chosen = sorted(
        candidates_with_ts,
        key=lambda x: (x["timestamp"], x["url_norm"], x["title_norm"]),
    )[0]
    return chosen, "ambiguous"


def _ensure_parent_dir(file_path):
    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _write_json(file_path, obj):
    _ensure_parent_dir(file_path)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def main():
    args = parse_args()

    with open(args.local_mashups_file, "r", encoding="utf-8") as f:
        local_mashups = json.load(f)

    active_rows = _fetch_json(args.active_mashups_url, args.timeout_sec)
    dead_rows = _fetch_json(args.dead_mashups_url, args.timeout_sec)

    source_entries = []
    for row in active_rows:
        if isinstance(row, dict):
            source_entries.append(_to_source_entry(row, "active"))
    for row in dead_rows:
        if isinstance(row, dict):
            source_entries.append(_to_source_entry(row, "deadpool"))

    title_index = defaultdict(list)
    url_index = defaultdict(list)
    for entry in source_entries:
        if entry["title_norm"]:
            title_index[entry["title_norm"]].append(entry)
        if entry["url_norm"]:
            url_index[entry["url_norm"]].append(entry)

    timestamp_map = {}
    unmatched = []
    method_counter = Counter()

    for mashup in local_mashups:
        mashup_id = int(mashup["id"])
        local_title_norm = normalize_title(mashup.get("title", ""))
        local_url_norm = normalize_url(mashup.get("url", ""))

        selected = None

        title_candidates = title_index.get(local_title_norm, [])
        if title_candidates:
            selected, mode = _select_candidate(title_candidates, local_url_norm)
            if selected is not None:
                method_counter[f"title:{mode}"] += 1

        if selected is None and local_url_norm:
            url_candidates = url_index.get(local_url_norm, [])
            if url_candidates:
                selected, mode = _select_candidate(url_candidates, local_url_norm)
                if selected is not None:
                    method_counter[f"url:{mode}"] += 1

        if selected is None:
            method_counter["unmatched"] += 1
            unmatched.append(
                {
                    "mashup_id": mashup_id,
                    "title": mashup.get("title", ""),
                    "url": mashup.get("url", ""),
                }
            )
            continue

        timestamp_map[str(mashup_id)] = float(selected["timestamp"])

    mapped_count = len(timestamp_map)
    total_local = len(local_mashups)
    coverage = float(mapped_count) / float(total_local) if total_local > 0 else 0.0

    source_with_ts = sum(1 for x in source_entries if x["timestamp"] is not None)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "local_mashups_file": args.local_mashups_file,
            "active_mashups_url": args.active_mashups_url,
            "dead_mashups_url": args.dead_mashups_url,
        },
        "source": {
            "active_rows": len(active_rows),
            "dead_rows": len(dead_rows),
            "total_rows": len(source_entries),
            "rows_with_parseable_timestamp": source_with_ts,
        },
        "mapping": {
            "local_total": total_local,
            "mapped": mapped_count,
            "unmapped": total_local - mapped_count,
            "coverage": coverage,
            "method_counts": dict(method_counter),
        },
    }

    _write_json(args.output_map_file, timestamp_map)
    _write_json(args.output_report_file, report)
    _write_json(args.output_unmatched_file, unmatched)

    print(
        "Timestamp map generated | "
        f"mapped={mapped_count}/{total_local} | coverage={coverage:.4f} | "
        f"map_file={args.output_map_file}"
    )
    print(f"Report file: {args.output_report_file}")
    print(f"Unmatched file: {args.output_unmatched_file}")


if __name__ == "__main__":
    main()
