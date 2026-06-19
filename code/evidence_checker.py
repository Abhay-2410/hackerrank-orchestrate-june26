"""Load and route minimum image evidence requirements."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from conversation_extractor import ExtractedClaim

ALWAYS_INCLUDE_APPLIES_TO = {"general claim review", "reviewability"}


def load_evidence(path: str) -> pd.DataFrame:
    """Load evidence requirements CSV and return a dataframe."""
    return pd.read_csv(Path(path), dtype=str).fillna("")


def _normalize_applies_to(value: str) -> str:
    return str(value).strip().lower()


def get_requirement(
    claim_object: str,
    df: pd.DataFrame,
    extracted: ExtractedClaim | None = None,
) -> str:
    """Return routed minimum_image_evidence text for the claim."""
    if extracted is None or not extracted.issue_families:
        mask = (df["claim_object"] == str(claim_object)) | (df["claim_object"] == "all")
        matches = df.loc[mask, "minimum_image_evidence"]
        if matches.empty:
            return ""
        return "\n".join(matches.tolist())

    target_families = {_normalize_applies_to(family) for family in extracted.issue_families}
    target_families.update(ALWAYS_INCLUDE_APPLIES_TO)
    if extracted.multi_image:
        target_families.add("multi-image rows")

    selected: list[str] = []
    seen: set[str] = set()

    for _, row in df.iterrows():
        row_object = str(row["claim_object"]).strip()
        if row_object not in {str(claim_object), "all"}:
            continue
        applies_to = _normalize_applies_to(row["applies_to"])
        if applies_to not in target_families:
            continue
        requirement_id = str(row["requirement_id"]).strip()
        text = str(row["minimum_image_evidence"]).strip()
        if not text or requirement_id in seen:
            continue
        seen.add(requirement_id)
        selected.append(f"[{requirement_id}] {text}")

    if not selected:
        mask = (df["claim_object"] == str(claim_object)) | (df["claim_object"] == "all")
        fallback = df.loc[mask, "minimum_image_evidence"]
        return "\n".join(fallback.tolist())

    return "\n".join(selected)


def get_matched_requirement_ids(
    claim_object: str,
    df: pd.DataFrame,
    extracted: ExtractedClaim | None = None,
) -> list[str]:
    """Return requirement_id values selected by the router (for audit trails)."""
    if extracted is None or not extracted.issue_families:
        mask = (df["claim_object"] == str(claim_object)) | (df["claim_object"] == "all")
        return df.loc[mask, "requirement_id"].astype(str).tolist()

    target_families = {_normalize_applies_to(family) for family in extracted.issue_families}
    target_families.update(ALWAYS_INCLUDE_APPLIES_TO)
    if extracted.multi_image:
        target_families.add("multi-image rows")

    ids: list[str] = []
    for _, row in df.iterrows():
        row_object = str(row["claim_object"]).strip()
        if row_object not in {str(claim_object), "all"}:
            continue
        applies_to = _normalize_applies_to(row["applies_to"])
        if applies_to in target_families:
            ids.append(str(row["requirement_id"]).strip())
    return list(dict.fromkeys(ids))
