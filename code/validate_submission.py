"""Pre-submission validator for HackerRank Orchestrate deliverables."""

from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = REPO_ROOT / "code"
LOG_PATH = Path.home() / "hackerrank_orchestrate" / "log.txt"

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

ALLOWED_CLAIM_OBJECTS = {"car", "laptop", "package"}
ALLOWED_CLAIM_STATUS = {"supported", "contradicted", "not_enough_information"}
ALLOWED_ISSUE_TYPES = {
    "dent", "scratch", "crack", "glass_shatter", "broken_part", "missing_part",
    "torn_packaging", "crushed_packaging", "water_damage", "stain", "none", "unknown",
}
ALLOWED_SEVERITY = {"none", "low", "medium", "high", "unknown"}
ALLOWED_RISK_FLAGS = {
    "none", "blurry_image", "cropped_or_obstructed", "low_light_or_glare", "wrong_angle",
    "wrong_object", "wrong_object_part", "damage_not_visible", "claim_mismatch",
    "possible_manipulation", "non_original_image", "text_instruction_present",
    "user_history_risk", "manual_review_required",
}
ALLOWED_PARTS = {
    "car": {
        "front_bumper", "rear_bumper", "door", "hood", "windshield", "side_mirror",
        "headlight", "taillight", "fender", "quarter_panel", "body", "unknown",
    },
    "laptop": {
        "screen", "keyboard", "trackpad", "hinge", "lid", "corner", "port", "base", "body", "unknown",
    },
    "package": {
        "box", "package_corner", "package_side", "seal", "label", "contents", "item", "unknown",
    },
}

ZIP_REQUIRED = [
    "code/main.py",
    "code/evaluation/main.py",
    "code/README.md",
    "code/requirements.txt",
    "code/claim_processor.py",
    "code/evaluation/evaluation_report.md",
]

ZIP_FORBIDDEN = (".env", "__pycache__", ".pyc", "sk-ant-api", "OLLAMA_API_KEY=")

_PLACEHOLDER_VALUES = frozenset({"...", "your-key-here", "your-ollama-key", "your-key"})

SECRET_PATTERNS = (
    re.compile(r"sk-ant-api\d+[A-Za-z0-9_-]{8,}", re.I),
    re.compile(r"OLLAMA_API_KEY=([^\s#\r\n]+)", re.I),
    re.compile(r"GEMINI_API_KEY=([^\s#\r\n]+)", re.I),
)


def _looks_like_real_secret(match: re.Match[str]) -> bool:
    """Ignore documentation placeholders and redacted log lines."""
    value = match.group(1) if match.lastindex else match.group(0)
    value = value.strip().strip('"').strip("'")
    if not value or value in _PLACEHOLDER_VALUES:
        return False
    if value.endswith("-here") or "your-" in value.lower():
        return False
    if "[REDACTED]" in value:
        return False
    return len(value) >= 12


class CheckResult:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.passed: list[str] = []

    def ok(self, message: str) -> None:
        self.passed.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def fail(self, message: str) -> None:
        self.errors.append(message)

    @property
    def success(self) -> bool:
        return not self.errors


def _check_output_csv(result: CheckResult) -> None:
    output_path = REPO_ROOT / "output.csv"
    claims_path = REPO_ROOT / "dataset" / "claims.csv"

    if not output_path.is_file():
        result.fail("Missing output.csv at repo root")
        return

    claims = pd.read_csv(claims_path, dtype=str).fillna("")
    output = pd.read_csv(output_path, dtype=str).fillna("")

    if list(output.columns) != OUTPUT_COLUMNS:
        result.fail(f"output.csv columns wrong. Expected {OUTPUT_COLUMNS}, got {list(output.columns)}")
    else:
        result.ok("output.csv has 14 columns in correct order")

    if len(output) != len(claims):
        result.fail(f"Row count mismatch: output={len(output)}, claims={len(claims)}")
    else:
        result.ok(f"output.csv has {len(output)} rows (matches claims.csv)")

    for index, row in output.iterrows():
        prefix = f"Row {index + 1} ({row.get('user_id', '?')})"

        if row.get("claim_object") not in ALLOWED_CLAIM_OBJECTS:
            result.fail(f"{prefix}: invalid claim_object={row.get('claim_object')!r}")

        for field in ("evidence_standard_met", "valid_image"):
            val = str(row.get(field, "")).strip().lower()
            if val not in {"true", "false"}:
                result.fail(f"{prefix}: {field} must be true/false, got {val!r}")

        if str(row.get("claim_status", "")).strip() not in ALLOWED_CLAIM_STATUS:
            result.fail(f"{prefix}: invalid claim_status={row.get('claim_status')!r}")

        if str(row.get("issue_type", "")).strip() not in ALLOWED_ISSUE_TYPES:
            result.fail(f"{prefix}: invalid issue_type={row.get('issue_type')!r}")

        if str(row.get("severity", "")).strip() not in ALLOWED_SEVERITY:
            result.fail(f"{prefix}: invalid severity={row.get('severity')!r}")

        obj = str(row.get("claim_object", "")).strip()
        part = str(row.get("object_part", "")).strip()
        if part not in ALLOWED_PARTS.get(obj, set()):
            result.fail(f"{prefix}: invalid object_part={part!r} for {obj}")

        flags_raw = str(row.get("risk_flags", "")).strip()
        if flags_raw.lower() != "none":
            for flag in flags_raw.split(";"):
                flag = flag.strip()
                if flag and flag not in ALLOWED_RISK_FLAGS:
                    result.fail(f"{prefix}: invalid risk_flag={flag!r}")

        ids_raw = str(row.get("supporting_image_ids", "")).strip()
        if ids_raw.lower() != "none":
            for image_id in ids_raw.split(";"):
                image_id = image_id.strip()
                if image_id and not re.match(r"^img_\d+$", image_id):
                    result.fail(f"{prefix}: invalid supporting_image_id={image_id!r}")

        justification = str(row.get("claim_status_justification", ""))
        if "Automated review failed" in justification or "No API key" in justification:
            result.fail(f"{prefix}: contains failure marker in justification")

        if len(str(row.get("claim_status_justification", ""))) > 500:
            result.warn(f"{prefix}: very long justification ({len(justification)} chars)")

        issue = str(row.get("issue_type", "")).strip()
        severity = str(row.get("severity", "")).strip()
        status = str(row.get("claim_status", "")).strip()
        if issue == "none" and severity != "none":
            result.fail(f"{prefix}: issue_type=none but severity={severity}")
        if status == "contradicted" and issue == "none" and severity != "none":
            result.fail(f"{prefix}: contradicted+none should have severity=none")

    failed = output[
        output["claim_status_justification"].str.contains("Automated review failed", na=False)
    ]
    if len(failed) == 0:
        result.ok("No failed/API-error rows in output.csv")
    else:
        result.fail(f"{len(failed)} rows contain Automated review failed")


def _check_code_zip(result: CheckResult) -> None:
    zip_path = REPO_ROOT / "code.zip"
    if not zip_path.is_file():
        result.fail("Missing code.zip at repo root")
        return

    size_kb = zip_path.stat().st_size / 1024
    result.ok(f"code.zip exists ({size_kb:.1f} KB)")

    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        for required in ZIP_REQUIRED:
            if required not in names:
                result.fail(f"code.zip missing required file: {required}")
            else:
                result.ok(f"code.zip contains {required}")

        if not any(name.startswith("code/evaluation/") for name in names):
            result.fail("code.zip missing evaluation/ folder")
        else:
            result.ok("code.zip contains evaluation/ folder")

        for name in names:
            lowered = name.lower()
            for forbidden in ZIP_FORBIDDEN:
                if forbidden in lowered or forbidden in archive.read(name).decode("utf-8", errors="ignore")[:5000]:
                    if forbidden in (".env", "__pycache__", ".pyc"):
                        if forbidden in lowered:
                            result.fail(f"code.zip contains forbidden path: {name}")
                    break


def _check_log(result: CheckResult) -> None:
    if not LOG_PATH.is_file():
        result.fail(f"Missing chat transcript: {LOG_PATH}")
        return

    text = LOG_PATH.read_text(encoding="utf-8", errors="ignore")
    lines = text.count("\n") + 1
    result.ok(f"Chat transcript exists ({lines} lines, {LOG_PATH})")

    if "AGREEMENT RECORDED" not in text and "SESSION START" not in text:
        result.warn("Log may be missing SESSION START / onboarding entries")

    for pattern in SECRET_PATTERNS:
        match = pattern.search(text)
        if match and _looks_like_real_secret(match):
            result.fail("Chat transcript appears to contain a raw API key — redact before upload")
            break
    else:
        result.ok("No raw API keys detected in chat transcript")


def _check_evaluation_report(result: CheckResult) -> None:
    report = CODE_DIR / "evaluation" / "evaluation_report.md"
    if not report.is_file():
        result.fail("Missing code/evaluation/evaluation_report.md")
        return

    text = report.read_text(encoding="utf-8")
    for section in ("Operational summary", "Strategy comparison", "Per-field accuracy"):
        if section.lower() not in text.lower():
            result.warn(f"evaluation_report.md may be missing section: {section}")
        else:
            result.ok(f"evaluation_report.md includes {section}")

    if "anthropic" not in text.lower():
        result.warn("evaluation_report.md does not mention Anthropic final strategy")


def _check_env_not_committed(result: CheckResult) -> None:
    env_path = REPO_ROOT / ".env"
    if env_path.is_file():
        result.ok(".env exists locally (should NOT be in zip or git)")
    gitignore = REPO_ROOT / ".gitignore"
    if gitignore.is_file() and ".env" in gitignore.read_text(encoding="utf-8"):
        result.ok(".env is listed in .gitignore")


def main() -> int:
    result = CheckResult()
    print("=" * 60)
    print("HackerRank Orchestrate — Final Submission Check")
    print("=" * 60)
    print()

    _check_output_csv(result)
    _check_code_zip(result)
    _check_log(result)
    _check_evaluation_report(result)
    _check_env_not_committed(result)

    print("PASSED:")
    for item in result.passed:
        print(f"  [OK] {item}")

    if result.warnings:
        print("\nWARNINGS:")
        for item in result.warnings:
            print(f"  [!] {item}")

    if result.errors:
        print("\nERRORS:")
        for item in result.errors:
            print(f"  [X] {item}")

    print()
    print("=" * 60)
    if result.success:
        print("RESULT: READY TO SUBMIT")
        print()
        print("Upload these three files on HackerRank:")
        print(f"  1. {REPO_ROOT / 'code.zip'}")
        print(f"  2. {REPO_ROOT / 'output.csv'}")
        print(f"  3. {LOG_PATH}")
    else:
        print(f"RESULT: NOT READY — fix {len(result.errors)} error(s) above")
    print("=" * 60)
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
