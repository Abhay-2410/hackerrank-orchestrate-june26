"""Offline post-processing to refine issue_type, severity, and evidence fields without extra API calls."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from conversation_extractor import ExtractedClaim


def _flags(result: dict[str, Any]) -> set[str]:
    raw = str(result.get("risk_flags", "none"))
    return {part.strip() for part in raw.split(";") if part.strip() and part.strip() != "none"}


def _primary_claimed_issue(extracted: ExtractedClaim | None) -> str | None:
    if not extracted or not extracted.claimed_issue_types:
        return None
    priority = (
        "crack",
        "glass_shatter",
        "dent",
        "scratch",
        "broken_part",
        "stain",
        "missing_part",
        "torn_packaging",
        "crushed_packaging",
        "water_damage",
    )
    for issue in priority:
        if issue in extracted.claimed_issue_types:
            return issue
    return extracted.claimed_issue_types[0]


def _conversation_text(user_claim: str) -> str:
    return re.sub(r"\s+", " ", user_claim.lower())


def apply_output_calibrator(
    result: dict[str, Any],
    claim_object: str,
    extracted: ExtractedClaim | None = None,
    user_claim: str = "",
) -> dict[str, Any]:
    """Apply conservative, high-confidence refinements after the severity engine."""
    text = _conversation_text(user_claim)
    flags = _flags(result)
    status = result["claim_status"]
    issue = result["issue_type"]
    part = result["object_part"]
    primary = _primary_claimed_issue(extracted)

    # Windshield chip/crack line is crack, not full glass_shatter
    if part == "windshield" or "windshield" in text or "front glass" in text:
        if issue == "glass_shatter" and any(
            word in text for word in ("crack", "chip", "spreading", "small stone", "line")
        ):
            if "shatter" not in text and "spider" not in text:
                result["issue_type"] = "crack"
                if result["severity"] == "high":
                    result["severity"] = "medium"

    # User claimed dent on bumper/panel — prefer dent over missing_part hallucination
    if issue == "missing_part" and "dent" in text and "missing" not in text:
        result["issue_type"] = "dent"
        if status == "supported" and result["severity"] == "high":
            result["severity"] = "medium"

    # Supported dent should not be high severity
    if issue == "dent" and status == "supported" and result["severity"] == "high":
        result["severity"] = "medium"

    # Minor laptop corner dent
    if claim_object == "laptop" and part == "corner" and issue == "dent" and status == "supported":
        if result["severity"] == "medium":
            result["severity"] = "low"

    # Contradicted scratch/mismatch on bumper
    if status == "contradicted" and "scratch" in text and issue == "dent":
        if "claim_mismatch" in flags or "pretty bad" in text or "severe" in text:
            result["issue_type"] = "scratch"
            result["severity"] = "low"

    # Multi-image identity failure on car scrape/damage claims
    if (
        claim_object == "car"
        and status == "contradicted"
        and "wrong_object" in flags
        and extracted
        and extracted.multi_image
        and str(result.get("supporting_image_ids", "none")).strip().lower() == "none"
    ):
        result["claim_status"] = "not_enough_information"
        result["issue_type"] = "broken_part"
        result["severity"] = "unknown"
        result["valid_image"] = True
        result["evidence_standard_met"] = False

    # Headlight claim but part not visible
    if part == "headlight" and status == "not_enough_information" and issue not in {"unknown", "none"}:
        if "damage_not_visible" in flags or "wrong_angle" in flags:
            result["issue_type"] = "unknown"
            result["severity"] = "unknown"

    # Images clear enough to evaluate even when claim is contradicted
    if status == "contradicted" and not result.get("evidence_standard_met"):
        if str(result.get("valid_image", "")).lower() == "true" or "non_original_image" in flags:
            result["evidence_standard_met"] = True
            if not str(result.get("evidence_standard_met_reason", "")).strip():
                result["evidence_standard_met_reason"] = (
                    "Images are sufficient to inspect the claimed part even though the claim is contradicted."
                )

    # Staged or wrong-content images stay invalid
    if "non_original_image" in flags and status == "contradicted":
        result["valid_image"] = False

    # Contents missing claims with unclear interior stay invalid
    if part == "contents" and status == "not_enough_information":
        if "damage_not_visible" in flags or "cropped_or_obstructed" in flags:
            result["valid_image"] = False

    return result
