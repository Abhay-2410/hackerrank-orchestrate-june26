"""Post-model consistency and severity calibration engine."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from conversation_extractor import ExtractedClaim
    from image_preflight import PreflightReport


def merge_risk_flags(existing: str, extra_flags: list[str]) -> str:
    """Merge semicolon-separated risk flags with additional flags."""
    current = [part.strip() for part in str(existing or "none").split(";") if part.strip()]
    if current == ["none"]:
        current = []
    for flag in extra_flags:
        if flag and flag not in current:
            current.append(flag)
    return ";".join(current) if current else "none"


def apply_severity_engine(
    result: dict[str, Any],
    claim_object: str,
    extracted: ExtractedClaim | None = None,
    preflight: PreflightReport | None = None,
) -> dict[str, Any]:
    """Apply cross-field consistency and severity calibration rules."""
    status = result["claim_status"]
    issue = result["issue_type"]
    severity = result["severity"]

    if issue == "none" and severity != "none":
        result["severity"] = "none"
    if status == "not_enough_information" and issue == "unknown" and severity not in {"unknown", "none"}:
        result["severity"] = "unknown"
    if status == "contradicted" and issue == "none" and severity != "none":
        result["severity"] = "none"
    if issue == "scratch" and severity == "high":
        result["severity"] = "low"
    if issue == "crack" and severity == "high" and claim_object == "car":
        result["severity"] = "medium"
    if issue == "glass_shatter" and severity in {"low", "medium"}:
        result["severity"] = "high"
    if issue in {"dent", "stain", "broken_part", "torn_packaging", "crushed_packaging", "water_damage"}:
        if severity == "unknown" and status == "supported":
            result["severity"] = "medium"
    if issue == "scratch" and severity == "unknown" and status == "supported":
        result["severity"] = "low"
    if issue == "missing_part" and status == "supported" and severity in {"unknown", "low"}:
        result["severity"] = "medium"
    if issue == "water_damage" and status == "supported" and severity in {"unknown", "low"}:
        result["severity"] = "medium"
    if status == "not_enough_information" and severity not in {"unknown", "none"}:
        if issue in {"unknown", "none"}:
            result["severity"] = "unknown"
    if status == "contradicted" and issue == "unknown" and severity not in {"unknown", "none"}:
        result["severity"] = "unknown"

    if extracted:
        if extracted.severity_claimed == "minor" and issue == "scratch" and status == "supported":
            result["severity"] = "low"
        if extracted.severity_claimed == "major" and issue == "glass_shatter" and status == "supported":
            result["severity"] = "high"
        if extracted.severity_claimed == "major" and status == "contradicted" and issue == "scratch":
            if "claim_mismatch" not in result["risk_flags"]:
                result["risk_flags"] = merge_risk_flags(result["risk_flags"], ["claim_mismatch"])

    if preflight and preflight.all_unusable:
        result["valid_image"] = False
        result["evidence_standard_met"] = False
        if result["claim_status"] != "contradicted":
            result["claim_status"] = "not_enough_information"
        result["severity"] = "unknown"
        result["issue_type"] = "unknown" if result["issue_type"] not in {"none"} else result["issue_type"]

    if preflight and preflight.suggested_risk_flags:
        result["risk_flags"] = merge_risk_flags(result["risk_flags"], preflight.suggested_risk_flags)

    if extracted and extracted.injection_detected:
        result["risk_flags"] = merge_risk_flags(result["risk_flags"], ["text_instruction_present"])

    flags = [part.strip() for part in result["risk_flags"].split(";") if part.strip()]
    if "manual_review_required" in flags and len(flags) > 1:
        flags = [flag for flag in flags if flag != "none"]
        result["risk_flags"] = ";".join(flags)

    if preflight and not preflight.all_unusable:
        blurry_only = (
            "blurry_image" in preflight.suggested_risk_flags
            and status == "supported"
            and result["severity"] in {"medium", "high", "unknown"}
        )
        if blurry_only and result["severity"] == "high" and issue not in {"glass_shatter"}:
            result["severity"] = "medium"

    return result
