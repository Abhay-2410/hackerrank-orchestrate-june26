"""Offline refinement and accuracy projection without API calls."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from claim_processor import OUTPUT_FIELDS, _post_process_output, _validate_model_output
from conversation_extractor import ExtractedClaim, extract_conversation
from image_preflight import PreflightReport, analyze_images


EVAL_FIELDS = [
    "claim_status",
    "evidence_standard_met",
    "issue_type",
    "severity",
    "valid_image",
]


def _normalize_bool(value: object) -> str:
    return str(value).strip().lower()


def _field_accuracy(expected_df: pd.DataFrame, predicted_df: pd.DataFrame, field: str) -> float:
    expected = expected_df[field].astype(str).str.strip()
    predicted = predicted_df[field].astype(str).str.strip()
    if field in {"evidence_standard_met", "valid_image"}:
        expected = expected.map(_normalize_bool)
        predicted = predicted.map(_normalize_bool)
    return float((expected == predicted).sum() / len(expected_df))


def _preflight_from_audit(audit: dict) -> PreflightReport | None:
    preflight_data = audit.get("preflight")
    if not isinstance(preflight_data, dict) or "images" not in preflight_data:
        return None
    report = PreflightReport()
    report.suggested_risk_flags = list(preflight_data.get("suggested_risk_flags", []))
    report.all_unusable = bool(preflight_data.get("all_unusable", False))
    report.summary_text = str(preflight_data.get("summary_text", ""))
    return report


def _extracted_from_audit(audit: dict) -> ExtractedClaim | None:
    data = audit.get("extracted_claim")
    if not isinstance(data, dict):
        return None
    return ExtractedClaim(**data)


def refine_row(row: pd.Series, result: dict, dataset_root: Path) -> dict[str, object]:
    """Re-run post-processing layers on an existing model result."""
    path_list = [part.strip() for part in str(row["image_paths"]).split(";") if part.strip()]
    extracted = extract_conversation(str(row["user_claim"]), str(row["claim_object"]), len(path_list))
    entries = [
        (Path(path).stem, (dataset_root / path.strip()).resolve()) for path in path_list
    ]
    preflight = analyze_images(entries)
    validated = _validate_model_output(
        {field: result.get(field) for field in OUTPUT_FIELDS},
        str(row["claim_object"]),
    )
    refined = _post_process_output(
        validated,
        str(row["claim_object"]),
        extracted,
        preflight,
        str(row["user_claim"]),
    )
    output = {col: row[col] for col in row.index if col not in OUTPUT_FIELDS}
    output.update(refined)
    for field in ("evidence_standard_met", "valid_image"):
        output[field] = _normalize_bool(output[field])
    return output


def refine_output_csv(
    claims_path: Path,
    input_csv: Path,
    output_csv: Path,
    dataset_root: Path,
) -> pd.DataFrame:
    claims_df = pd.read_csv(claims_path, dtype=str).fillna("")
    current_df = pd.read_csv(input_csv, dtype=str).fillna("")
    refined_rows: list[dict] = []
    for index, claim_row in claims_df.iterrows():
        result = current_df.iloc[index].to_dict()
        refined_rows.append(refine_row(claim_row, result, dataset_root))
    refined_df = pd.DataFrame(refined_rows)
    refined_df.to_csv(output_csv, index=False)
    return refined_df


def project_sample_accuracy(audit_dir: Path, sample_path: Path) -> dict[str, float]:
    sample_df = pd.read_csv(sample_path, dtype=str).fillna("")
    audits_by_user: dict[str, dict] = {}
    for audit_path in audit_dir.glob("*.json"):
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        audits_by_user[str(audit["user_id"])] = audit

    predicted_rows: list[dict] = []
    for _, sample_row in sample_df.iterrows():
        user_id = str(sample_row["user_id"])
        audit = audits_by_user.get(user_id, {})
        source = audit.get("primary_output") or audit.get("final_output") or {}
        extracted = _extracted_from_audit(audit) or extract_conversation(
            str(sample_row["user_claim"]),
            str(sample_row["claim_object"]),
            len(str(sample_row["image_paths"]).split(";")),
        )
        preflight = _preflight_from_audit(audit)
        validated = _validate_model_output(source, str(sample_row["claim_object"]))
        refined = _post_process_output(
            validated,
            str(sample_row["claim_object"]),
            extracted,
            preflight,
            str(sample_row["user_claim"]),
        )
        row = sample_row.to_dict()
        row.update(refined)
        for field in ("evidence_standard_met", "valid_image"):
            row[field] = _normalize_bool(row[field])
        predicted_rows.append(row)
    predicted_df = pd.DataFrame(predicted_rows)
    return {field: _field_accuracy(sample_df, predicted_df, field) for field in EVAL_FIELDS}


def project_sample_accuracy_from_outputs(
    sample_path: Path,
    outputs: list[dict],
) -> dict[str, float]:
    sample_df = pd.read_csv(sample_path, dtype=str).fillna("")
    predicted_df = pd.DataFrame(outputs)
    return {field: _field_accuracy(sample_df, predicted_df, field) for field in EVAL_FIELDS}


def main() -> None:
    dataset_root = REPO_ROOT / "dataset"
    audit_dir = CODE_DIR / "evaluation" / "audit"
    sample_path = dataset_root / "sample_claims.csv"
    input_csv = REPO_ROOT / "output.csv"
    output_csv = REPO_ROOT / "output.csv"

    before = project_sample_accuracy(audit_dir, sample_path) if audit_dir.is_dir() else {}

    # Measure "before calibrator" baseline from cached primary outputs
    baseline_rows: list[dict] = []
    if audit_dir.is_dir():
        audits_by_user: dict[str, dict] = {}
        sample_df = pd.read_csv(sample_path, dtype=str).fillna("")
        for audit_path in audit_dir.glob("*.json"):
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            audits_by_user[str(audit["user_id"])] = audit
        for _, sample_row in sample_df.iterrows():
            audit = audits_by_user.get(str(sample_row["user_id"]), {})
            source = audit.get("final_output") or audit.get("primary_output") or {}
            row = sample_row.to_dict()
            row.update(source)
            for field in ("evidence_standard_met", "valid_image"):
                if field in row:
                    row[field] = _normalize_bool(row[field])
            baseline_rows.append(row)
        before = project_sample_accuracy_from_outputs(sample_path, baseline_rows)

    refine_output_csv(dataset_root / "claims.csv", input_csv, output_csv, dataset_root)
    after = project_sample_accuracy(audit_dir, sample_path) if audit_dir.is_dir() else {}

    print("Offline refinement complete (0 API calls).")
    print(f"Updated: {output_csv}")
    if before and after:
        print("\nProjected sample-set accuracy from cached Anthropic outputs:")
        print(f"{'Field':<25} {'Before':>10} {'After':>10}")
        print("-" * 47)
        for field in EVAL_FIELDS:
            print(f"{field:<25} {before[field]:>9.1%} {after[field]:>9.1%}")


if __name__ == "__main__":
    main()
