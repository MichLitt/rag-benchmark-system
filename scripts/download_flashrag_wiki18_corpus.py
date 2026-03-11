from __future__ import annotations

import argparse
import gzip
import json
import zipfile
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download FlashRAG wiki18 retrieval corpus and normalize to passages.jsonl.gz."
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="RUC-NLPIR/FlashRAG_datasets",
        help="HuggingFace dataset repo id.",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="retrieval-corpus/wiki18_100w.zip",
        help="File path in the dataset repo.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/corpus/wiki18_100w"),
        help="Directory to save downloaded and normalized artifacts.",
    )
    parser.add_argument(
        "--normalized-name",
        type=str,
        default="passages.jsonl.gz",
        help="Output normalized filename under output-dir.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Maximum rows to normalize; <=0 means full corpus.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download even if file exists in local cache.",
    )
    parser.add_argument(
        "--skip-normalize",
        action="store_true",
        help="Only download zip, skip normalization.",
    )
    return parser.parse_args()


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _split_contents(contents: str) -> tuple[str, str]:
    if "[SEP]" not in contents:
        raw = contents.strip()
        if "\n" in raw:
            maybe_title, maybe_text = raw.split("\n", 1)
            title = maybe_title.strip().strip('"').strip()
            text = maybe_text.strip()
            if title and text:
                return title, text
        return "", raw
    title, text = contents.split("[SEP]", 1)
    return title.strip(), text.strip()


def _find_jsonl_member(zip_path: Path) -> str:
    with zipfile.ZipFile(zip_path, "r") as zf:
        candidates = [m.filename for m in zf.infolist() if not m.is_dir() and m.filename.endswith(".jsonl")]
    if not candidates:
        raise ValueError(f"No .jsonl file found in archive: {zip_path}")
    if len(candidates) > 1:
        print(f"Found multiple jsonl files, using first: {candidates[0]}")
    return candidates[0]


def _normalize_row(row: dict[str, Any]) -> dict[str, str]:
    doc_id = str(row.get("doc_id") or row.get("id") or "")
    title = str(row.get("title") or "").strip()
    text = str(row.get("text") or "").strip()
    contents = str(row.get("contents") or "").strip()

    if contents and (not text or "[SEP]" in contents):
        parsed_title, parsed_text = _split_contents(contents)
        if parsed_title:
            title = parsed_title
        if parsed_text:
            text = parsed_text

    return {"doc_id": doc_id, "title": title, "text": text}


def _normalize_zip_jsonl(zip_path: Path, member_name: str, out_path: Path, max_rows: int) -> int:
    count = 0
    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open(member_name, "r") as src, gzip.open(out_path, "wt", encoding="utf-8") as dst:
            for raw in src:
                line = raw.decode("utf-8").strip()
                if not line:
                    continue
                row = json.loads(line)
                normalized = _normalize_row(row)
                dst.write(json.dumps(normalized, ensure_ascii=False) + "\n")
                count += 1
                if count % 100_000 == 0:
                    print(f"Normalized {count} passages...")
                if max_rows > 0 and count >= max_rows:
                    break
    return count


def main() -> None:
    args = parse_args()
    _ensure_dir(args.output_dir)

    print(f"Downloading {args.repo_id}:{args.filename} ...")
    zip_path = Path(
        hf_hub_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            filename=args.filename,
            local_dir=str(args.output_dir),
            force_download=bool(args.force_download),
        )
    )
    print(f"Downloaded zip path: {zip_path}")

    normalized_path = args.output_dir / args.normalized_name
    rows = 0
    archive_member = ""
    if not args.skip_normalize:
        archive_member = _find_jsonl_member(zip_path)
        print(f"Normalizing member: {archive_member}")
        rows = _normalize_zip_jsonl(
            zip_path=zip_path,
            member_name=archive_member,
            out_path=normalized_path,
            max_rows=int(args.max_rows),
        )
        print(f"Done. Normalized {rows} rows to {normalized_path}")

    manifest = {
        "source_repo": args.repo_id,
        "source_filename": args.filename,
        "zip_path": str(zip_path),
        "archive_member": archive_member,
        "normalized_path": str(normalized_path) if not args.skip_normalize else "",
        "rows": rows,
        "max_rows": int(args.max_rows),
    }
    manifest_path = args.output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"Manifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()
