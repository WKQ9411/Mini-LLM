import argparse
import json
import math
from pathlib import Path
import pandas as pd


def _extract_overall_score(rating: object) -> float | None:
    if not isinstance(rating, dict):
        return None

    raw_score = rating.get("overall_score")
    try:
        score = float(raw_score)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(score):
        return None
    return score


def _normalize_gap_to_score(gap: float | None, min_gap: float, max_gap: float) -> int:
    if gap is None or not math.isfinite(gap):
        return 3

    if max_gap <= min_gap:
        return 3

    normalized_gap = (gap - min_gap) / (max_gap - min_gap)
    score = int(round(1 + normalized_gap * 4))
    return max(1, min(5, score))


def convert(input_path: str, output_path: str) -> None:
    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df)} rows from {input_path}")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    prepared_rows: list[dict] = []
    skipped = 0

    for idx, row in df.iterrows():
        prompt = str(row.get("instruction", "")).strip()
        chosen = str(row.get("chosen_response", "")).strip()
        rejected = str(row.get("rejected_response", "")).strip()

        if not prompt or not chosen or not rejected:
            skipped += 1
            continue

        chosen_score = _extract_overall_score(row.get("chosen_rating"))
        rejected_score = _extract_overall_score(row.get("rejected_rating"))
        gap = None if chosen_score is None or rejected_score is None else chosen_score - rejected_score

        prepared_rows.append(
            {
                "sample_id": idx,
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "gap": gap,
            }
        )

    valid_gaps = [item["gap"] for item in prepared_rows if item["gap"] is not None]
    if valid_gaps:
        min_gap = min(valid_gaps)
        max_gap = max(valid_gaps)
    else:
        min_gap = 0.0
        max_gap = 0.0

    written = 0
    fallback_scores = 0
    with open(output, "w", encoding="utf-8") as f:
        for item in prepared_rows:
            score = _normalize_gap_to_score(item["gap"], min_gap, max_gap)
            if item["gap"] is None:
                fallback_scores += 1

            record = {
                "sample_id": item["sample_id"],
                "prompt": item["prompt"],
                "chosen": item["chosen"],
                "rejected": item["rejected"],
                "score": score,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"Done. Written: {written}, Skipped: {skipped}, Fallback scores: {fallback_scores}")
    print(f"Output: {output}")


def main():
    root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(description="Convert parquet DPO data to jsonl")
    parser.add_argument(
        "--input",
        type=str,
        default=str(root / "data/dpo_data/ultrafeedback_zh_binarized_lowest.parquet"),
        help="Input parquet file path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(root / "data/dpo_data/dpo.jsonl"),
        help="Output jsonl file path",
    )
    args = parser.parse_args()

    convert(args.input, args.output)


if __name__ == "__main__":
    main()
