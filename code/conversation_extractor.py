"""Extract structured claim understanding from user conversation text."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field


INJECTION_PATTERNS = (
    r"approve\s+immediately",
    r"ignore\s+(all\s+)?instructions",
    r"mark\s+(it\s+)?supported",
    r"bypass\s+review",
    r"do\s+not\s+check",
    r"override\s+system",
)

ISSUE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "dent": ("dent", "dented", "deformation", "bulge", "creased panel"),
    "scratch": ("scratch", "scrape", "scuffed", "scuff", "mark", "scratched"),
    "crack": ("crack", "cracked", "chip", "spreading crack", "fracture"),
    "glass_shatter": ("shatter", "shattered", "spider", "spider-web", "smashed glass"),
    "broken_part": ("broken", "broke", "snapped", "detached", "smashed", "scrape lag gaya", "scrape"),
    "missing_part": ("missing part", "part missing", "item missing", "empty box", "not inside", "gone"),
    "torn_packaging": ("torn", "ripped", "opened flap", "seal broken", "tape torn"),
    "crushed_packaging": ("crushed", "crumpled", "squashed", "dented box", "corner crush"),
    "water_damage": ("water", "wet", "moisture", "soaked", "rain damage"),
    "stain": ("stain", "stained", "liquid spill", "coffee", "spill"),
}

PART_KEYWORDS: dict[str, dict[str, tuple[str, ...]]] = {
    "car": {
        "front_bumper": ("front bumper", "front side", "front end", "bumper ke upar"),
        "rear_bumper": ("rear bumper", "back bumper", "back of the car", "rear end"),
        "door": ("door", "car door"),
        "hood": ("hood", "bonnet"),
        "windshield": ("windshield", "front glass", "wind screen", "front windshield"),
        "side_mirror": ("mirror", "side mirror", "wing mirror"),
        "headlight": ("headlight", "head light"),
        "taillight": ("taillight", "tail light", "rear light"),
        "fender": ("fender",),
        "quarter_panel": ("quarter panel",),
        "body": ("body", "panel", "car body"),
    },
    "laptop": {
        "screen": ("screen", "display", "lcd", "monitor"),
        "keyboard": ("keyboard", "keys", "key"),
        "trackpad": ("trackpad", "touchpad", "mouse pad"),
        "hinge": ("hinge",),
        "lid": ("lid", "cover", "top cover"),
        "corner": ("corner",),
        "port": ("port", "usb", "charging port"),
        "base": ("base", "bottom"),
        "body": ("body", "chassis", "case"),
    },
    "package": {
        "box": ("box", "carton", "package"),
        "package_corner": ("corner", "edge crush"),
        "package_side": ("side", "panel"),
        "seal": ("seal", "tape", "flap"),
        "label": ("label", "shipping label"),
        "contents": ("contents", "inside", "inner", "items inside"),
        "item": ("item", "product", "phone", "device"),
    },
}

ISSUE_TO_EVIDENCE_FAMILY: dict[str, str] = {
    "dent": "dent or scratch",
    "scratch": "dent or scratch",
    "crack": "crack, broken, or missing part",
    "glass_shatter": "crack, broken, or missing part",
    "broken_part": "crack, broken, or missing part",
    "missing_part": "contents or inner item",
    "torn_packaging": "crushed, torn, or seal damage",
    "crushed_packaging": "crushed, torn, or seal damage",
    "water_damage": "water, stain, or label damage",
    "stain": "water, stain, or label damage",
}

OBJECT_DEFAULT_FAMILIES: dict[str, list[str]] = {
    "car": ["general claim review", "vehicle identity or orientation"],
    "laptop": ["general claim review", "screen, keyboard, or trackpad", "hinge, lid, corner, body, or port"],
    "package": ["general claim review", "crushed, torn, or seal damage"],
}


@dataclass
class ExtractedClaim:
    """Structured understanding extracted from the claim conversation."""

    claimed_issue_types: list[str] = field(default_factory=list)
    claimed_parts: list[str] = field(default_factory=list)
    severity_claimed: str = "unknown"
    issue_families: list[str] = field(default_factory=list)
    multi_image: bool = False
    injection_detected: bool = False
    language_hint: str = "en"

    def to_dict(self) -> dict:
        return asdict(self)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _find_keywords(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)


def extract_conversation(user_claim: str, claim_object: str, image_count: int = 1) -> ExtractedClaim:
    """Parse conversation for claimed issues, parts, and evidence routing hints."""
    text = _normalize(user_claim)
    result = ExtractedClaim(multi_image=image_count > 1)

    if re.search(r"[\u0900-\u097F]", user_claim):
        result.language_hint = "hi-en" if re.search(r"[a-zA-Z]", user_claim) else "hi"

    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text):
            result.injection_detected = True
            break

    for issue_type, keywords in ISSUE_KEYWORDS.items():
        if _find_keywords(text, keywords):
            result.claimed_issue_types.append(issue_type)

    part_map = PART_KEYWORDS.get(claim_object, {})
    for part, keywords in part_map.items():
        if _find_keywords(text, keywords):
            result.claimed_parts.append(part)

    if any(word in text for word in ("minor", "small", "light", "slight", "little", "thoda")):
        result.severity_claimed = "minor"
    elif any(
        word in text
        for word in ("major", "severe", "bad", "totaled", "destroyed", "heavy", "big damage")
    ):
        result.severity_claimed = "major"

    families: list[str] = list(OBJECT_DEFAULT_FAMILIES.get(claim_object, ["general claim review"]))
    if result.multi_image:
        families.append("multi-image rows")
    families.append("reviewability")

    for issue in result.claimed_issue_types:
        family = ISSUE_TO_EVIDENCE_FAMILY.get(issue)
        if family and family not in families:
            families.append(family)

    if claim_object == "car" and any(
        word in text for word in ("parked", "side", "orientation", "identity", "front", "rear", "back")
    ):
        if "vehicle identity or orientation" not in families:
            families.append("vehicle identity or orientation")

    if claim_object == "package" and any(word in text for word in ("inside", "contents", "missing item", "empty")):
        if "contents or inner item" not in families:
            families.append("contents or inner item")

    if claim_object == "laptop" and any(part in result.claimed_parts for part in ("hinge", "lid", "corner", "port", "base", "body")):
        if "hinge, lid, corner, body, or port" not in families:
            families.append("hinge, lid, corner, body, or port")

    result.issue_families = list(dict.fromkeys(families))
    return result
