"""Confidence scoring and escalation decisions for the signature stack."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from conversation_extractor import ExtractedClaim
    from image_preflight import PreflightReport

CONFIDENCE_ESCALATION_THRESHOLD = float(os.environ.get("ESCALATION_THRESHOLD", "0.62"))


def compute_confidence(
    result: dict[str, Any],
    extracted: ExtractedClaim | None,
    preflight: PreflightReport | None,
    *,
    parse_retries: int = 0,
    provider: str = "",
) -> float:
    """Return a 0-1 confidence score for the current claim result."""
    score = 1.0

    if parse_retries > 0:
        score -= 0.15 * parse_retries

    if result.get("claim_status") == "not_enough_information":
        score -= 0.08

    if result.get("severity") == "unknown":
        score -= 0.12

    if result.get("issue_type") == "unknown":
        score -= 0.08

    flags = {
        part.strip()
        for part in str(result.get("risk_flags", "none")).split(";")
        if part.strip() and part.strip() != "none"
    }
    high_risk_flags = {
        "claim_mismatch",
        "possible_manipulation",
        "non_original_image",
        "wrong_object",
        "wrong_object_part",
        "manual_review_required",
        "text_instruction_present",
    }
    score -= 0.06 * len(flags & high_risk_flags)
    score -= 0.03 * len(flags - high_risk_flags)

    if preflight:
        if preflight.all_unusable:
            score -= 0.25
        elif preflight.suggested_risk_flags:
            score -= 0.04 * len(preflight.suggested_risk_flags)

    if extracted:
        if extracted.injection_detected:
            score -= 0.1
        if extracted.claimed_issue_types and result.get("issue_type") not in extracted.claimed_issue_types:
            if result.get("claim_status") == "supported":
                score -= 0.1
        if extracted.claimed_parts and result.get("object_part") not in extracted.claimed_parts:
            if result.get("object_part") != "unknown":
                score -= 0.05

    if not result.get("evidence_standard_met") and result.get("claim_status") == "supported":
        score -= 0.15

    if provider and provider != "anthropic":
        score -= 0.03

    return max(0.0, min(1.0, score))


def should_escalate(
    confidence: float,
    result: dict[str, Any],
    provider: str,
    anthropic_available: bool,
) -> bool:
    """Decide whether to run a verification call on a stronger model."""
    if not anthropic_available:
        return False
    if provider == "anthropic":
        return False
    if confidence >= CONFIDENCE_ESCALATION_THRESHOLD:
        return False

    flags = {
        part.strip()
        for part in str(result.get("risk_flags", "none")).split(";")
        if part.strip() and part.strip() != "none"
    }
    if "manual_review_required" in flags:
        return True
    if result.get("severity") == "unknown" and result.get("claim_status") == "supported":
        return True
    if "claim_mismatch" in flags:
        return True
    if confidence < 0.5:
        return True
    return confidence < CONFIDENCE_ESCALATION_THRESHOLD


def build_verification_prompt(
    user_claim: str,
    claim_object: str,
    draft: dict[str, Any],
    extracted: ExtractedClaim | None,
) -> str:
    """Build a narrow second-pass prompt to verify uncertain results."""
    extracted_text = extracted.to_dict() if extracted else {}
    return f"""You are verifying an uncertain damage-claim review. Images remain the primary source of truth.

CLAIM OBJECT: {claim_object}
USER CONVERSATION:
{user_claim}

STRUCTURED CLAIM UNDERSTANDING:
{extracted_text}

DRAFT DECISION TO VERIFY:
{draft}

Re-inspect all images. Confirm or correct ONLY these fields:
evidence_standard_met, evidence_standard_met_reason, risk_flags, issue_type, object_part,
claim_status, claim_status_justification, supporting_image_ids, valid_image, severity

Respond ONLY with valid JSON using the same allowed enum values as the draft.
Be conservative when images are ambiguous. Short strings under 180 chars.
Required JSON keys:
evidence_standard_met, evidence_standard_met_reason, risk_flags, issue_type, object_part,
claim_status, claim_status_justification, supporting_image_ids, valid_image, severity
"""
