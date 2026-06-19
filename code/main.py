"""Run claim predictions on dataset/claims.csv and write output.csv."""

import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from claim_processor import (
    get_active_provider,
    get_usage_stats,
    normalize_supporting_image_ids,
    process_claim,
    reset_usage_stats,
    set_dataset_base_path,
)
from evidence_checker import load_evidence
from history_lookup import load_history

DATASET_BASE_PATH = "../dataset"

OUTPUT_COLUMNS = [
    "user_id",
    "image_paths",
    "user_claim",
    "claim_object",
    "evidence_standard_met",
    "evidence_standard_met_reason",
    "risk_flags",
    "issue_type",
    "object_part",
    "claim_status",
    "claim_status_justification",
    "supporting_image_ids",
    "valid_image",
    "severity",
]


def _resolve_dataset_path(relative: str) -> Path:
    return (CODE_DIR / DATASET_BASE_PATH / relative).resolve()


def _is_failed_result(row: dict) -> bool:
    reason = str(row.get("evidence_standard_met_reason", ""))
    justification = str(row.get("claim_status_justification", ""))
    markers = (
        "429",
        "Automated review failed",
        "No API key",
        "RESOURCE_EXHAUSTED",
        "application/octet-stream",
        "invalid image",
    )
    return any(marker in reason or marker in justification for marker in markers)


def _write_run_stats(
    claim_count: int,
    failed_count: int,
    elapsed_seconds: float,
    output_path: Path,
) -> None:
    stats = get_usage_stats()
    provider, model = get_active_provider()
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "provider": stats.get("provider") or provider,
        "model": stats.get("model") or model,
        "test_claims": claim_count,
        "failed_claims": failed_count,
        "api_calls": stats["api_calls"],
        "primary_calls": stats.get("primary_calls", stats["api_calls"]),
        "verify_calls": stats.get("verify_calls", 0),
        "escalations": stats.get("escalations", 0),
        "input_tokens": stats["input_tokens"],
        "output_tokens": stats["output_tokens"],
        "images_processed": stats["images_processed"],
        "elapsed_seconds": round(elapsed_seconds, 1),
        "seconds_per_claim": round(elapsed_seconds / max(claim_count, 1), 2),
        "output_csv": str(output_path),
        "pipeline": "signature_stack",
    }
    stats_path = CODE_DIR / "evaluation" / "run_stats.json"
    stats_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


async def _process_all_claims(
    claims_df: pd.DataFrame,
    history_df: pd.DataFrame,
    evidence_df: pd.DataFrame,
    concurrency: int = 3,
) -> list[dict]:
    semaphore = asyncio.Semaphore(concurrency)
    rows = [row for _, row in claims_df.iterrows()]
    results: list = [None] * len(rows)

    async def _run(index: int, row: pd.Series) -> None:
        async with semaphore:
            results[index] = await process_claim(row, history_df, evidence_df)

    await asyncio.gather(*[_run(index, row) for index, row in enumerate(rows)])
    return results


async def _run_with_retries(
    claims_df: pd.DataFrame,
    history_df: pd.DataFrame,
    evidence_df: pd.DataFrame,
) -> list[dict]:
    results = await _process_all_claims(claims_df, history_df, evidence_df, concurrency=3)
    rows = [row for _, row in claims_df.iterrows()]

    for pass_num in range(2):
        failed_indexes = [i for i, result in enumerate(results) if _is_failed_result(result)]
        if not failed_indexes:
            break
        print(f"Retry pass {pass_num + 1}: {len(failed_indexes)} failed claims...")
        await asyncio.sleep(5)
        retry_df = pd.DataFrame([rows[i] for i in failed_indexes])
        retried = await _process_all_claims(retry_df, history_df, evidence_df, concurrency=2)
        for idx, new_result in zip(failed_indexes, retried):
            if not _is_failed_result(new_result):
                results[idx] = new_result
    return results


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    reset_usage_stats()
    started = time.time()

    dataset_root = (CODE_DIR / DATASET_BASE_PATH).resolve()
    set_dataset_base_path(str(dataset_root))

    history_df = load_history(_resolve_dataset_path("user_history.csv"))
    evidence_df = load_evidence(_resolve_dataset_path("evidence_requirements.csv"))
    claims_df = pd.read_csv(_resolve_dataset_path("claims.csv"), dtype=str).fillna("")

    provider, model = get_active_provider()
    if provider == "none":
        print("Warning: no API key configured. Set ANTHROPIC_API_KEY, OLLAMA_API_KEY, or GEMINI_API_KEY in .env")
    else:
        print(f"Using provider: {provider} / {model}")

    results = asyncio.run(_run_with_retries(claims_df, history_df, evidence_df))
    elapsed = time.time() - started
    output_df = pd.DataFrame(results)

    for column in OUTPUT_COLUMNS:
        if column not in output_df.columns:
            output_df[column] = ""

    output_df = output_df[OUTPUT_COLUMNS]
    output_df["evidence_standard_met"] = output_df["evidence_standard_met"].map(
        lambda value: str(value).lower() if str(value).lower() in {"true", "false"} else str(value)
    )
    output_df["valid_image"] = output_df["valid_image"].map(
        lambda value: str(value).lower() if str(value).lower() in {"true", "false"} else str(value)
    )
    output_df["supporting_image_ids"] = output_df["supporting_image_ids"].map(normalize_supporting_image_ids)

    output_path = REPO_ROOT / "output.csv"
    output_df.to_csv(output_path, index=False)

    failed_count = sum(1 for row in results if _is_failed_result(row))
    _write_run_stats(len(output_df), failed_count, elapsed, output_path)

    stats = get_usage_stats()
    print(f"Wrote {len(output_df)} rows to {output_path}")
    print(
        f"Provider: {stats.get('provider', 'none')}, model: {stats.get('model', 'none')}, "
        f"API calls: {stats['api_calls']}, "
        f"input tokens: {stats['input_tokens']}, "
        f"output tokens: {stats['output_tokens']}, "
        f"images processed: {stats['images_processed']}, "
        f"failed: {failed_count}, "
        f"runtime: {elapsed:.0f}s"
    )
    print(f"Test run stats saved to {CODE_DIR / 'evaluation' / 'run_stats.json'}")


if __name__ == "__main__":
    main()
