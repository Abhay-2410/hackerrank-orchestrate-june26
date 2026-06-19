"""Evaluate claim predictions against dataset/sample_claims.csv."""

import asyncio
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

EVAL_DIR = Path(__file__).resolve().parent
CODE_DIR = EVAL_DIR.parent
REPO_ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from claim_processor import (
    clear_audit_trails,
    get_active_provider,
    get_audit_trails,
    get_usage_stats,
    process_claim,
    reset_usage_stats,
    set_dataset_base_path,
)
from evidence_checker import load_evidence
from history_lookup import load_history

DATASET_BASE_PATH = "../../dataset"
TEST_CLAIM_COUNT = 44

EVAL_FIELDS = [
    "claim_status",
    "evidence_standard_met",
    "issue_type",
    "severity",
    "valid_image",
]

INPUT_PRICE_PER_MTOK = 3.0
OUTPUT_PRICE_PER_MTOK = 15.0

STRATEGY_COMPARISON = [
    {
        "name": "Gemini 2.5-flash-lite (free tier)",
        "claim_status": 0.50,
        "evidence_standard_met": 0.60,
        "issue_type": 0.75,
        "severity": 0.60,
        "valid_image": 0.80,
        "notes": "Hit 429 quota limits on full test set.",
    },
    {
        "name": "Ollama gemma4:31b (free tier)",
        "claim_status": None,
        "evidence_standard_met": None,
        "issue_type": None,
        "severity": None,
        "valid_image": None,
        "notes": "Free cloud vision model; current run populates accuracy below.",
    },
    {
        "name": "Anthropic claude-sonnet-4-6 (recommended final)",
        "claim_status": None,
        "evidence_standard_met": None,
        "issue_type": None,
        "severity": None,
        "valid_image": None,
        "notes": "Set ANTHROPIC_API_KEY in .env for highest precision (auto-selected when present).",
    },
]


def _resolve_dataset_path(relative: str) -> Path:
    return (EVAL_DIR / DATASET_BASE_PATH / relative).resolve()


def _normalize_bool(value: object) -> str:
    return str(value).strip().lower()


async def _process_all_claims(
    claims_df: pd.DataFrame,
    history_df: pd.DataFrame,
    evidence_df: pd.DataFrame,
) -> list[dict]:
    semaphore = asyncio.Semaphore(3)
    rows = [row for _, row in claims_df.iterrows()]
    results: list = [None] * len(rows)

    async def _run(index: int, row: pd.Series) -> None:
        async with semaphore:
            results[index] = await process_claim(
                row, history_df, evidence_df, collect_audit=True
            )

    await asyncio.gather(*[_run(index, row) for index, row in enumerate(rows)])
    return results


def _field_accuracy(expected_df: pd.DataFrame, predicted_df: pd.DataFrame, field: str) -> float:
    expected = expected_df[field].astype(str).str.strip()
    predicted = predicted_df[field].astype(str).str.strip()
    if field in {"evidence_standard_met", "valid_image"}:
        expected = expected.map(_normalize_bool)
        predicted = predicted.map(_normalize_bool)
    matches = (expected == predicted).sum()
    total = len(expected_df)
    return (matches / total) if total else 0.0


def _confusion_matrix(
    expected_df: pd.DataFrame,
    predicted_df: pd.DataFrame,
    field: str,
) -> dict[str, dict[str, int]]:
    labels = sorted(set(expected_df[field].astype(str)) | set(predicted_df[field].astype(str)))
    matrix = {label: {other: 0 for other in labels} for label in labels}
    for expected, predicted in zip(expected_df[field], predicted_df[field]):
        matrix[str(expected).strip()][str(predicted).strip()] += 1
    return matrix


def _accuracy_by_object(
    expected_df: pd.DataFrame,
    predicted_df: pd.DataFrame,
    field: str,
) -> dict[str, float]:
    breakdown: dict[str, float] = {}
    for claim_object in sorted(expected_df["claim_object"].unique()):
        mask = expected_df["claim_object"] == claim_object
        if mask.sum() == 0:
            continue
        breakdown[str(claim_object)] = _field_accuracy(
            expected_df.loc[mask].reset_index(drop=True),
            predicted_df.loc[mask].reset_index(drop=True),
            field,
        )
    return breakdown


def _failure_rows(
    expected_df: pd.DataFrame,
    predicted_df: pd.DataFrame,
    fields: list[str],
    limit: int = 5,
) -> list[str]:
    notes: list[str] = []
    for index in range(len(expected_df)):
        mismatches: list[str] = []
        for field in fields:
            expected_val = expected_df.iloc[index][field]
            predicted_val = predicted_df.iloc[index][field]
            if field in {"evidence_standard_met", "valid_image"}:
                matched = _normalize_bool(expected_val) == _normalize_bool(predicted_val)
            else:
                matched = str(expected_val).strip() == str(predicted_val).strip()
            if not matched:
                mismatches.append(field)
        if not mismatches:
            continue
        user_id = expected_df.iloc[index]["user_id"]
        claim_object = expected_df.iloc[index]["claim_object"]
        detail = ", ".join(
            f"{field}: expected={expected_df.iloc[index][field]!s}, "
            f"got={predicted_df.iloc[index][field]!s}"
            for field in mismatches
        )
        notes.append(f"{user_id} ({claim_object}) — {detail}")
        if len(notes) >= limit:
            break
    return notes


def _write_audit_files(audit_dir: Path, audits: list[dict]) -> None:
    audit_dir.mkdir(parents=True, exist_ok=True)
    for index, audit in enumerate(audits, start=1):
        user_id = str(audit.get("user_id", f"case_{index}"))
        safe_name = re.sub(r"[^\w.-]+", "_", user_id)
        path = audit_dir / f"{index:02d}_{safe_name}.json"
        path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")


def _estimate_test_cost(provider: str, tokens_per_claim: int, test_claims: int = TEST_CLAIM_COUNT) -> str:
    if provider == "anthropic":
        input_tok = tokens_per_claim * 0.7
        output_tok = tokens_per_claim * 0.3
        cost = (test_claims * input_tok / 1_000_000) * INPUT_PRICE_PER_MTOK + (
            test_claims * output_tok / 1_000_000
        ) * OUTPUT_PRICE_PER_MTOK
        return f"~${cost:.2f} for {test_claims} test claims at Sonnet 4.6 pricing"
    if provider == "ollama":
        return "$0 (Ollama cloud free tier assumed)"
    return "$0 (Gemini free tier assumed)"


def _load_test_run_stats() -> dict | None:
    stats_path = EVAL_DIR / "run_stats.json"
    if not stats_path.is_file():
        return None
    try:
        return json.loads(stats_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _tpm_rpm_notes(provider: str, tokens_per_claim: int, seconds_per_claim: float) -> list[str]:
    calls_per_minute = 60 / max(seconds_per_claim, 0.1)
    tokens_per_minute = tokens_per_claim * calls_per_minute * 3  # concurrency 3
    if provider == "anthropic":
        return [
            f"- Anthropic Sonnet tier limits vary by account; at 3 concurrent claims ~{calls_per_minute * 3:.0f} RPM",
            f"- Estimated peak TPM ~{tokens_per_minute:,.0f} (3 parallel × ~{tokens_per_claim:,} tok/claim)",
            "- Backoff: exponential retry on 429/529 (up to 3 attempts)",
        ]
    if provider == "ollama":
        return [
            f"- Ollama cloud: ~{calls_per_minute:.0f} claims/min at observed latency; 0.5s delay between calls",
            f"- Estimated throughput ~{tokens_per_minute:,.0f} TPM with concurrency 3",
            "- Backoff: exponential retry on 429/503",
        ]
    return [
        f"- Gemini free tier: strict RPM/TPM; sequential fallback recommended if 429 persists",
        f"- Estimated ~{tokens_per_minute:,.0f} TPM at concurrency 3",
        "- Backoff: exponential retry on 429",
    ]


def _write_report(
    report_path: Path,
    total_claims: int,
    accuracies: dict[str, float],
    stats: dict,
    elapsed_seconds: float,
    notes: list[str],
    test_run: dict | None,
    claim_status_matrix: dict[str, dict[str, int]],
    severity_matrix: dict[str, dict[str, int]],
    severity_by_object: dict[str, float],
    escalations: int,
) -> None:
    provider = stats.get("provider", "unknown")
    model = stats.get("model", "unknown")
    active_provider, active_model = get_active_provider()
    if provider == "unknown" and active_provider != "none":
        provider, model = active_provider, active_model

    if provider == "anthropic":
        pricing_note = f"Sonnet 4.6 @ ${INPUT_PRICE_PER_MTOK}/MTok in, ${OUTPUT_PRICE_PER_MTOK}/MTok out"
        total_cost = (stats["input_tokens"] / 1_000_000) * INPUT_PRICE_PER_MTOK + (
            stats["output_tokens"] / 1_000_000
        ) * OUTPUT_PRICE_PER_MTOK
    else:
        pricing_note = "Ollama/Gemini free tier (assumed $0)"
        total_cost = 0.0

    tokens_per_claim = int(
        (stats["input_tokens"] + stats["output_tokens"]) / max(stats["api_calls"], 1)
    )
    avg_seconds = elapsed_seconds / max(total_claims, 1)

    lines = [
        "# Evaluation Report",
        "",
        f"Total sample claims evaluated: {total_claims}",
        "",
        "## Signature stack architecture",
        "",
        "Pipeline: conversation extraction → image pre-flight CV → evidence requirement routing →",
        "primary vision LLM → severity/consistency engine → confidence gate → optional Anthropic verification.",
        "",
        f"- Primary model calls: {stats.get('primary_calls', stats['api_calls'])}",
        f"- Verification escalations: {escalations}",
        f"- Verify model calls: {stats.get('verify_calls', 0)}",
        "",
        "## Per-field accuracy (current strategy)",
        "",
        "| Field | Accuracy |",
        "|---|---:|",
    ]
    for field, accuracy in accuracies.items():
        lines.append(f"| {field} | {accuracy:.1%} |")

    lines.extend(["", "## claim_status confusion matrix", ""])
    status_labels = sorted(claim_status_matrix.keys())
    lines.append("| expected \\ predicted | " + " | ".join(status_labels) + " |")
    lines.append("|" + "---|" * (len(status_labels) + 1))
    for expected in status_labels:
        row = [expected]
        for predicted in status_labels:
            row.append(str(claim_status_matrix[expected].get(predicted, 0)))
        lines.append("| " + " | ".join(row) + " |")

    lines.extend(["", "## severity confusion matrix", ""])
    severity_labels = sorted(severity_matrix.keys())
    lines.append("| expected \\ predicted | " + " | ".join(severity_labels) + " |")
    lines.append("|" + "---|" * (len(severity_labels) + 1))
    for expected in severity_labels:
        row = [expected]
        for predicted in severity_labels:
            row.append(str(severity_matrix[expected].get(predicted, 0)))
        lines.append("| " + " | ".join(row) + " |")

    lines.extend(["", "## severity accuracy by claim_object", ""])
    for claim_object, accuracy in severity_by_object.items():
        lines.append(f"- {claim_object}: {accuracy:.1%}")

    lines.extend(
        [
            "",
            "## Strategy comparison",
            "",
            "| Strategy | claim_status | evidence_standard_met | issue_type | severity | valid_image |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for strategy in STRATEGY_COMPARISON:
        name = strategy["name"]
        is_current = (
            (provider == "anthropic" and "Anthropic" in name)
            or (provider == "ollama" and "Ollama" in name)
            or (provider == "gemini" and "Gemini" in name)
        )
        if strategy["claim_status"] is None and is_current:
            cs = f"{accuracies.get('claim_status', 0):.1%}"
            es = f"{accuracies.get('evidence_standard_met', 0):.1%}"
            it = f"{accuracies.get('issue_type', 0):.1%}"
            se = f"{accuracies.get('severity', 0):.1%}"
            vi = f"{accuracies.get('valid_image', 0):.1%}"
        elif strategy["claim_status"] is None:
            cs = es = it = se = vi = "pending"
        else:
            cs = f"{strategy['claim_status']:.1%}"
            es = f"{strategy['evidence_standard_met']:.1%}"
            it = f"{strategy['issue_type']:.1%}"
            se = f"{strategy['severity']:.1%}"
            vi = f"{strategy['valid_image']:.1%}"
        lines.append(f"| {name} | {cs} | {es} | {it} | {se} | {vi} |")

    final_provider = test_run.get("provider", provider) if test_run else provider
    final_model = test_run.get("model", model) if test_run else model
    lines.extend(
        [
            "",
            f"**Final strategy for output.csv:** {final_provider} / {final_model}",
            "",
            "## Operational summary (sample set)",
            "",
            f"- Provider / model: {provider} / {model}",
            f"- Approximate API calls (sample): {stats['api_calls']}",
            f"- Approximate input tokens: {stats['input_tokens']:,}",
            f"- Approximate output tokens: {stats['output_tokens']:,}",
            f"- Images processed: {stats['images_processed']}",
            f"- Approximate cost ({pricing_note}): ${total_cost:.4f}",
            f"- Sample evaluation runtime: {elapsed_seconds:.0f}s ({avg_seconds:.1f}s per claim)",
            f"- Estimated full test runtime: {avg_seconds * TEST_CLAIM_COUNT:.0f}s at same concurrency",
            f"- Estimated full test cost: {_estimate_test_cost(provider, tokens_per_claim)}",
        ]
    )

    if test_run:
        test_provider = test_run.get("provider", "unknown")
        test_model = test_run.get("model", "unknown")
        test_elapsed = float(test_run.get("elapsed_seconds", 0))
        test_failed = int(test_run.get("failed_claims", 0))
        lines.extend(
            [
                "",
                "## Test set run (latest — from code/main.py)",
                "",
                f"- Timestamp (UTC): {test_run.get('timestamp_utc', 'unknown')}",
                f"- Provider / model: {test_provider} / {test_model}",
                f"- Test claims processed: {test_run.get('test_claims', TEST_CLAIM_COUNT)}",
                f"- Failed claims: {test_failed}",
                f"- API calls: {test_run.get('api_calls', 0)}",
                f"- Input tokens: {int(test_run.get('input_tokens', 0)):,}",
                f"- Output tokens: {int(test_run.get('output_tokens', 0)):,}",
                f"- Images processed: {test_run.get('images_processed', 0)}",
                f"- Runtime: {test_elapsed:.0f}s ({test_run.get('seconds_per_claim', 0)}s per claim)",
                f"- Output file: `{test_run.get('output_csv', 'output.csv')}`",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## Test set run",
                "",
                "- No `run_stats.json` yet. Run `python code/main.py` to generate test predictions and stats.",
            ]
        )

    lines.extend(
        [
            "",
            "## Rate limits, TPM/RPM, and batching",
            "",
            "- Concurrency: 3 parallel claims (`asyncio.Semaphore(3)`)",
            "- Retry: up to 3 API retries on 429/503; JSON parse retry once per claim",
            "- Main.py auto-retries failed claims up to 2 passes",
            "- Delay: 0.5s between successful Ollama/Gemini calls",
            "- Images re-encoded to JPEG via Pillow before upload",
            "- Signature stack: extract → pre-flight → route evidence → primary VLM → severity engine → escalate if low confidence",
            "- Escalation: uncertain claims verified with Anthropic when ANTHROPIC_API_KEY is set and primary is not Anthropic",
            "- Audit JSON written to `code/evaluation/audit/` during sample evaluation",
            "",
        ]
    )
    lines.extend(_tpm_rpm_notes(provider, tokens_per_claim, avg_seconds))
    lines.extend(["", "## Notes", ""])
    if notes:
        lines.extend(f"- {note}" for note in notes)
    else:
        lines.append("- No notable failure patterns recorded.")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    reset_usage_stats()
    clear_audit_trails()
    started = time.time()

    dataset_root = (EVAL_DIR / DATASET_BASE_PATH).resolve()
    set_dataset_base_path(str(dataset_root))

    history_df = load_history(_resolve_dataset_path("user_history.csv"))
    evidence_df = load_evidence(_resolve_dataset_path("evidence_requirements.csv"))
    sample_df = pd.read_csv(_resolve_dataset_path("sample_claims.csv"), dtype=str).fillna("")

    input_columns = ["user_id", "image_paths", "user_claim", "claim_object"]
    predictions = asyncio.run(
        _process_all_claims(sample_df[input_columns], history_df, evidence_df)
    )
    elapsed = time.time() - started
    predicted_df = pd.DataFrame(predictions)

    accuracies = {
        field: _field_accuracy(sample_df, predicted_df, field) for field in EVAL_FIELDS
    }
    claim_status_matrix = _confusion_matrix(sample_df, predicted_df, "claim_status")
    severity_matrix = _confusion_matrix(sample_df, predicted_df, "severity")
    severity_by_object = _accuracy_by_object(sample_df, predicted_df, "severity")

    print("Per-field accuracy:")
    print(f"{'Field':<25} {'Accuracy':>10}")
    print("-" * 37)
    for field, accuracy in accuracies.items():
        print(f"{field:<25} {accuracy:>9.1%}")

    stats = get_usage_stats()
    audits = get_audit_trails()
    audit_dir = EVAL_DIR / "audit"
    _write_audit_files(audit_dir, audits)
    notes: list[str] = []

    if stats["api_calls"] == 0:
        notes.append(
            "No successful API calls were recorded; check OLLAMA_API_KEY, GEMINI_API_KEY, "
            "or ANTHROPIC_API_KEY and image availability."
        )
    if stats["images_processed"] < len(sample_df):
        notes.append(
            "Some sample images were missing on disk, which typically forces not_enough_information outcomes."
        )

    failure_notes = _failure_rows(sample_df, predicted_df, EVAL_FIELDS, limit=5)
    if failure_notes:
        notes.append("Top failure patterns on sample set:")
        notes.extend(failure_notes)

    worst_field = min(accuracies, key=accuracies.get)
    if accuracies[worst_field] < 1.0:
        notes.append(
            f"Lowest accuracy on '{worst_field}' ({accuracies[worst_field]:.1%}); "
            "severity engine and Anthropic escalation target this gap."
        )
    notes.append(
        "Refinement: add ANTHROPIC_API_KEY to .env for primary or escalation verification, "
        "then run `scripts/refine.ps1` (or eval + main manually)."
    )

    report_path = EVAL_DIR / "evaluation_report.md"
    test_run = _load_test_run_stats()
    _write_report(
        report_path,
        len(sample_df),
        accuracies,
        stats,
        elapsed,
        notes,
        test_run,
        claim_status_matrix,
        severity_matrix,
        severity_by_object,
        int(stats.get("escalations", 0)),
    )
    print(f"\nWrote evaluation report to {report_path}")
    print(f"Wrote {len(audits)} audit files to {audit_dir}")


if __name__ == "__main__":
    main()
