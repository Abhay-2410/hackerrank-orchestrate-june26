"""Process a single damage claim via a vision LLM API (Ollama, Gemini, or Anthropic)."""

import asyncio
import base64
import json
import os
import re
from pathlib import Path
from typing import Any, Optional

import anthropic

from confidence import build_verification_prompt, compute_confidence, should_escalate
from conversation_extractor import extract_conversation
from evidence_checker import get_matched_requirement_ids, get_requirement
from history_lookup import get_history
from image_preflight import analyze_images
from prompt_builder import ALLOWED_OBJECT_PARTS, build_prompt
from severity_engine import apply_severity_engine

ANTHROPIC_MODEL = "claude-sonnet-4-6"
GEMINI_MODEL = "gemini-2.5-flash-lite"
DEFAULT_OLLAMA_MODEL = "gemma4:31b"
DEFAULT_OLLAMA_HOST = "https://ollama.com"
MAX_RETRIES = 3
API_CALL_DELAY_SEC = 0.5

ALLOWED_CLAIM_STATUS = {"supported", "contradicted", "not_enough_information"}
ALLOWED_ISSUE_TYPES = {
    "dent",
    "scratch",
    "crack",
    "glass_shatter",
    "broken_part",
    "missing_part",
    "torn_packaging",
    "crushed_packaging",
    "water_damage",
    "stain",
    "none",
    "unknown",
}
ALLOWED_RISK_FLAGS = {
    "none",
    "blurry_image",
    "cropped_or_obstructed",
    "low_light_or_glare",
    "wrong_angle",
    "wrong_object",
    "wrong_object_part",
    "damage_not_visible",
    "claim_mismatch",
    "possible_manipulation",
    "non_original_image",
    "text_instruction_present",
    "user_history_risk",
    "manual_review_required",
}
ALLOWED_SEVERITY = {"none", "low", "medium", "high", "unknown"}

OUTPUT_FIELDS = [
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

_dataset_base_path: Optional[str] = None
_usage_stats = {
    "api_calls": 0,
    "input_tokens": 0,
    "output_tokens": 0,
    "images_processed": 0,
    "provider": "",
    "model": "",
    "escalations": 0,
    "primary_calls": 0,
    "verify_calls": 0,
}

_audit_trails: list[dict[str, Any]] = []


def set_dataset_base_path(path: str) -> None:
    """Set the dataset root used to resolve image_paths from CSV rows."""
    global _dataset_base_path
    _dataset_base_path = path


def get_usage_stats() -> dict[str, int | str]:
    """Return cumulative API usage counters for evaluation reporting."""
    return dict(_usage_stats)


def reset_usage_stats() -> None:
    """Reset API usage counters."""
    _usage_stats["api_calls"] = 0
    _usage_stats["input_tokens"] = 0
    _usage_stats["output_tokens"] = 0
    _usage_stats["images_processed"] = 0
    _usage_stats["provider"] = ""
    _usage_stats["model"] = ""
    _usage_stats["escalations"] = 0
    _usage_stats["primary_calls"] = 0
    _usage_stats["verify_calls"] = 0


def clear_audit_trails() -> None:
    """Clear collected per-claim audit records."""
    _audit_trails.clear()


def get_audit_trails() -> list[dict[str, Any]]:
    """Return audit records collected during the latest run."""
    return list(_audit_trails)


def _resolve_api_config() -> tuple[str, str, str, str]:
    """Return (provider, api_key, model, host). Anthropic preferred when configured."""
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if anthropic_key and "your-actual-key" not in anthropic_key and "your_key" not in anthropic_key:
        return "anthropic", anthropic_key, ANTHROPIC_MODEL, ""

    ollama_key = os.environ.get("OLLAMA_API_KEY", "").strip()
    if ollama_key:
        model = os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL).strip() or DEFAULT_OLLAMA_MODEL
        host = os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST).strip() or DEFAULT_OLLAMA_HOST
        return "ollama", ollama_key, model, host

    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if gemini_key and gemini_key.strip():
        return "gemini", gemini_key.strip(), GEMINI_MODEL, ""

    return "", "", "", ""


def _resolve_anthropic_config() -> tuple[str, str]:
    """Return Anthropic (api_key, model) when configured for escalation."""
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if anthropic_key and "your-actual-key" not in anthropic_key and "your_key" not in anthropic_key:
        return anthropic_key, ANTHROPIC_MODEL
    return "", ""


def _resolve_image_path(relative_path: str) -> Path:
    if _dataset_base_path is None:
        raise RuntimeError("dataset base path not configured; call set_dataset_base_path first")
    return (Path(_dataset_base_path) / relative_path.strip()).resolve()


def _image_id_from_path(path: str) -> str:
    return Path(path.strip()).stem


def _encode_image_jpeg(image_path: Path) -> Optional[str]:
    """Load image and re-encode as JPEG bytes so APIs receive valid image/jpeg data."""
    if not image_path.is_file():
        return None
    try:
        import io

        from PIL import Image

        with Image.open(image_path) as img:
            rgb = img.convert("RGB")
            buffer = io.BytesIO()
            rgb.save(buffer, format="JPEG", quality=92)
            return base64.standard_b64encode(buffer.getvalue()).decode("ascii")
    except Exception:
        return None


def _strip_markdown_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, count=1)
        cleaned = re.sub(r"\s*```$", "", cleaned, count=1)
    return cleaned.strip()


def _parse_json_response(text: str) -> dict[str, Any]:
    """Parse model JSON with light repair for common formatting issues."""
    cleaned = _strip_markdown_fences(text)
    attempts = [cleaned]
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        attempts.append(match.group())
    for candidate in attempts:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            repaired = re.sub(r",\s*([}\]])", r"\1", candidate)
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                continue
    raise json.JSONDecodeError("Could not parse model JSON response", cleaned, 0)


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def _validate_risk_flags(value: Any) -> str:
    if value is None or str(value).strip() == "":
        return "manual_review_required"
    raw = str(value).strip()
    if raw.lower() == "none":
        return "none"
    flags = [part.strip() for part in raw.split(";") if part.strip()]
    valid = [flag for flag in flags if flag in ALLOWED_RISK_FLAGS]
    if not valid:
        return "manual_review_required"
    if "none" in valid and len(valid) > 1:
        valid = [flag for flag in valid if flag != "none"]
    return ";".join(valid) if valid else "none"


def _validate_object_part(value: Any, claim_object: str) -> str:
    allowed = set(ALLOWED_OBJECT_PARTS.get(claim_object, ["unknown"]))
    part = str(value or "unknown").strip()
    return part if part in allowed else "unknown"


def _validate_choice(value: Any, allowed: set[str], default: str) -> str:
    choice = str(value or default).strip()
    return choice if choice in allowed else default


def _validate_supporting_image_ids(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, list):
        ids = [str(part).strip().strip("'\"") for part in value if str(part).strip()]
        valid = [part for part in ids if re.match(r"^img_\d+$", part)]
        return ";".join(valid) if valid else "none"
    raw = str(value).strip()
    if not raw or raw.lower() in {"none", "[]"}:
        return "none"
    found = re.findall(r"img_\d+", raw)
    if found:
        return ";".join(dict.fromkeys(found))
    ids = [part.strip().strip("'\"") for part in raw.split(";") if part.strip()]
    valid = [part for part in ids if part and part.lower() != "none"]
    return ";".join(valid) if valid else "none"


def normalize_supporting_image_ids(value: Any) -> str:
    """Public helper to normalize supporting_image_ids to img_1;img_2 or none."""
    return _validate_supporting_image_ids(value)


def _validate_model_output(parsed: dict[str, Any], claim_object: str) -> dict[str, Any]:
    return {
        "evidence_standard_met": _parse_bool(parsed.get("evidence_standard_met", False)),
        "evidence_standard_met_reason": str(
            parsed.get("evidence_standard_met_reason", "Unable to determine evidence sufficiency.")
        ).strip(),
        "risk_flags": _validate_risk_flags(parsed.get("risk_flags")),
        "issue_type": _validate_choice(parsed.get("issue_type"), ALLOWED_ISSUE_TYPES, "unknown"),
        "object_part": _validate_object_part(parsed.get("object_part"), claim_object),
        "claim_status": _validate_choice(
            parsed.get("claim_status"), ALLOWED_CLAIM_STATUS, "not_enough_information"
        ),
        "claim_status_justification": str(
            parsed.get(
                "claim_status_justification",
                "Insufficient information to produce a grounded justification.",
            )
        ).strip(),
        "supporting_image_ids": _validate_supporting_image_ids(parsed.get("supporting_image_ids")),
        "valid_image": _parse_bool(parsed.get("valid_image", False)),
        "severity": _validate_choice(parsed.get("severity"), ALLOWED_SEVERITY, "unknown"),
    }


def get_active_provider() -> tuple[str, str]:
    """Return (provider, model) for the configured API without making a call."""
    provider, _, model, _ = _resolve_api_config()
    return provider or "none", model or "none"


def _fallback_output(reason: str) -> dict[str, Any]:
    return {
        "evidence_standard_met": False,
        "evidence_standard_met_reason": reason,
        "risk_flags": "manual_review_required",
        "issue_type": "unknown",
        "object_part": "unknown",
        "claim_status": "not_enough_information",
        "claim_status_justification": reason,
        "supporting_image_ids": "none",
        "valid_image": False,
        "severity": "unknown",
    }


def _build_anthropic_content(image_entries: list[tuple[str, str]], prompt: str) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for index, (image_id, encoded) in enumerate(image_entries, start=1):
        content.append({"type": "text", "text": f"Image {index} ({image_id})"})
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": encoded,
                },
            }
        )
    content.append({"type": "text", "text": prompt})
    return content


def _build_ollama_messages(image_entries: list[tuple[str, str]], prompt: str) -> list[dict[str, Any]]:
    labels = [f"Image {index} ({image_id})" for index, (image_id, _) in enumerate(image_entries, start=1)]
    content = "\n".join(labels) + "\n\n" + prompt
    images = [encoded for _, encoded in image_entries]
    return [{"role": "user", "content": content, "images": images}]


def _build_gemini_parts(image_entries: list[tuple[str, str]], prompt: str) -> list[Any]:
    from google.genai import types

    parts: list[Any] = []
    for index, (image_id, encoded) in enumerate(image_entries, start=1):
        parts.append(types.Part.from_text(text=f"Image {index} ({image_id})"))
        parts.append(
            types.Part.from_bytes(
                data=base64.standard_b64decode(encoded),
                mime_type="image/jpeg",
            )
        )
    parts.append(types.Part.from_text(text=prompt))
    return parts


def _record_usage(provider: str, model: str, input_tokens: int = 0, output_tokens: int = 0) -> None:
    _usage_stats["api_calls"] += 1
    _usage_stats["provider"] = provider
    _usage_stats["model"] = model
    _usage_stats["input_tokens"] += input_tokens
    _usage_stats["output_tokens"] += output_tokens


def _is_rate_limit_error(exc: Exception) -> bool:
    message = str(exc).lower()
    if any(
        token in message
        for token in ("429", "503", "resource exhausted", "rate limit", "unavailable", "quota")
    ):
        return True
    status_code = getattr(exc, "status_code", None)
    return status_code in {429, 503, 529}


async def _call_ollama(api_key: str, model: str, host: str, messages: list[dict[str, Any]]) -> str:
    from ollama import Client

    client = Client(host=host, headers={"Authorization": f"Bearer {api_key}"})
    last_error: Optional[Exception] = None
    for attempt in range(MAX_RETRIES):
        try:
            response = await asyncio.to_thread(
                client.chat,
                model=model,
                messages=messages,
                stream=False,
            )
            input_tokens = int(getattr(response, "prompt_eval_count", 0) or 0)
            output_tokens = int(getattr(response, "eval_count", 0) or 0)
            _record_usage("ollama", model, input_tokens, output_tokens)
            await asyncio.sleep(API_CALL_DELAY_SEC)
            return response.message.content or ""
        except Exception as exc:
            if _is_rate_limit_error(exc):
                last_error = exc
            else:
                raise
        if attempt < MAX_RETRIES - 1:
            await asyncio.sleep((2**attempt) * 3)
    raise last_error or RuntimeError("Ollama API call failed after retries")


async def _call_anthropic(api_key: str, model: str, content: list[dict[str, Any]]) -> str:
    client = anthropic.AsyncAnthropic(api_key=api_key)
    last_error: Optional[Exception] = None
    for attempt in range(MAX_RETRIES):
        try:
            response = await client.messages.create(
                model=model,
                max_tokens=2048,
                temperature=0,
                system=(
                    "You are a damage-claim evidence reviewer. "
                    "Respond with a single valid JSON object only — no markdown, no prose outside JSON."
                ),
                messages=[{"role": "user", "content": content}],
            )
            input_tokens = response.usage.input_tokens if response.usage else 0
            output_tokens = response.usage.output_tokens if response.usage else 0
            _record_usage("anthropic", model, input_tokens or 0, output_tokens or 0)
            text_blocks = [block.text for block in response.content if block.type == "text"]
            return "".join(text_blocks)
        except anthropic.RateLimitError as exc:
            last_error = exc
        except anthropic.APIStatusError as exc:
            if exc.status_code in {429, 529}:
                last_error = exc
            else:
                raise
        if attempt < MAX_RETRIES - 1:
            await asyncio.sleep(2**attempt)
    raise last_error or RuntimeError("Anthropic API call failed after retries")


def _call_gemini_sync(api_key: str, model: str, parts: list[Any]) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=parts,
        config=types.GenerateContentConfig(
            max_output_tokens=1024,
            temperature=0.1,
            response_mime_type="application/json",
        ),
    )
    input_tokens = 0
    output_tokens = 0
    if response.usage_metadata:
        input_tokens = response.usage_metadata.prompt_token_count or 0
        output_tokens = response.usage_metadata.candidates_token_count or 0
    _record_usage("gemini", model, input_tokens, output_tokens)
    return response.text or ""


async def _call_gemini(api_key: str, model: str, parts: list[Any]) -> str:
    last_error: Optional[Exception] = None
    for attempt in range(MAX_RETRIES):
        try:
            result = await asyncio.to_thread(_call_gemini_sync, api_key, model, parts)
            await asyncio.sleep(API_CALL_DELAY_SEC)
            return result
        except Exception as exc:
            if _is_rate_limit_error(exc):
                last_error = exc
            else:
                raise
        if attempt < MAX_RETRIES - 1:
            delay = (2**attempt) * 2
            await asyncio.sleep(delay)
    raise last_error or RuntimeError("Gemini API call failed after retries")


async def _call_vision_model(
    provider: str,
    api_key: str,
    model: str,
    host: str,
    image_entries: list[tuple[str, str]],
    prompt: str,
    *,
    call_kind: str = "primary",
) -> str:
    """Invoke the configured vision provider with a prompt and encoded images."""
    if call_kind == "primary":
        _usage_stats["primary_calls"] += 1
    elif call_kind == "verify":
        _usage_stats["verify_calls"] += 1

    if provider == "ollama":
        messages = _build_ollama_messages(image_entries, prompt)
        return await _call_ollama(api_key, model, host, messages)
    if provider == "gemini":
        parts = _build_gemini_parts(image_entries, prompt)
        return await _call_gemini(api_key, model, parts)
    content = _build_anthropic_content(image_entries, prompt)
    return await _call_anthropic(api_key, model, content)


async def _run_model_pass(
    provider: str,
    api_key: str,
    model: str,
    host: str,
    image_entries: list[tuple[str, str]],
    prompt: str,
    claim_object: str,
    *,
    call_kind: str = "primary",
) -> tuple[dict[str, Any], int]:
    """Run one model pass with JSON parse retry; return validated output and parse retries."""
    last_error: Optional[Exception] = None
    parse_retries = 0
    for parse_attempt in range(2):
        try:
            raw_response = await _call_vision_model(
                provider,
                api_key,
                model,
                host,
                image_entries,
                prompt,
                call_kind=call_kind,
            )
            parsed = _parse_json_response(raw_response)
            parsed.pop("_reasoning", None)
            return _validate_model_output(parsed, claim_object), parse_retries
        except json.JSONDecodeError as exc:
            last_error = exc
            parse_retries += 1
            if parse_attempt == 0:
                await asyncio.sleep(1)
                continue
        except Exception as exc:
            last_error = exc
            break
    raise last_error or RuntimeError("Model pass failed")


async def process_claim(
    row: Any,
    history_df: Any,
    evidence_df: Any,
    *,
    collect_audit: bool = False,
) -> dict[str, Any]:
    """Analyze one claim row through the signature stack and return all 14 output columns."""
    user_id = str(row["user_id"])
    image_paths_raw = str(row["image_paths"])
    user_claim = str(row["user_claim"])
    claim_object = str(row["claim_object"])

    path_list = [part.strip() for part in image_paths_raw.split(";") if part.strip()]
    extracted = extract_conversation(user_claim, claim_object, image_count=len(path_list))
    user_history = get_history(user_id, history_df)
    evidence_requirement = get_requirement(claim_object, evidence_df, extracted)
    requirement_ids = get_matched_requirement_ids(claim_object, evidence_df, extracted)

    preflight_entries: list[tuple[str, Path]] = []
    image_entries: list[tuple[str, str]] = []
    missing_images: list[str] = []

    for relative_path in path_list:
        resolved = _resolve_image_path(relative_path)
        image_id = _image_id_from_path(relative_path)
        preflight_entries.append((image_id, resolved))
        encoded = _encode_image_jpeg(resolved)
        if encoded is None:
            missing_images.append(relative_path)
            continue
        image_entries.append((image_id, encoded))
        _usage_stats["images_processed"] += 1

    preflight = analyze_images(preflight_entries)
    prompt = build_prompt(
        user_claim,
        claim_object,
        evidence_requirement,
        user_history,
        extracted_claim=extracted.to_dict(),
        preflight_summary=preflight.summary_text,
        preflight_flags=preflight.suggested_risk_flags,
    )

    base_result = {
        "user_id": user_id,
        "image_paths": image_paths_raw,
        "user_claim": user_claim,
        "claim_object": claim_object,
    }
    audit: dict[str, Any] = {
        "user_id": user_id,
        "claim_object": claim_object,
        "extracted_claim": extracted.to_dict(),
        "matched_requirement_ids": requirement_ids,
        "preflight": preflight.to_dict() if collect_audit else {"suggested_risk_flags": preflight.suggested_risk_flags},
    }

    if not image_entries:
        reason = "Submitted image files were missing or unreadable."
        if missing_images:
            reason = f"Could not load submitted images: {', '.join(missing_images)}."
        validated = apply_severity_engine(
            _validate_model_output(_fallback_output(reason), claim_object),
            claim_object,
            extracted,
            preflight,
        )
        if collect_audit:
            audit["confidence"] = 0.0
            audit["escalated"] = False
            audit["final_output"] = {key: validated[key] for key in OUTPUT_FIELDS}
            _audit_trails.append(audit)
        return {**base_result, **validated}

    provider, api_key, model, host = _resolve_api_config()
    if not api_key:
        validated = apply_severity_engine(
            _validate_model_output(
                _fallback_output(
                    "No API key configured. Set OLLAMA_API_KEY, GEMINI_API_KEY, or ANTHROPIC_API_KEY in .env."
                ),
                claim_object,
            ),
            claim_object,
            extracted,
            preflight,
        )
        if collect_audit:
            audit["confidence"] = 0.0
            audit["escalated"] = False
            audit["final_output"] = {key: validated[key] for key in OUTPUT_FIELDS}
            _audit_trails.append(audit)
        return {**base_result, **validated}

    parse_retries = 0
    try:
        validated, parse_retries = await _run_model_pass(
            provider,
            api_key,
            model,
            host,
            image_entries,
            prompt,
            claim_object,
            call_kind="primary",
        )
    except Exception as exc:
        validated = _validate_model_output(
            _fallback_output(f"Automated review failed: {exc}"),
            claim_object,
        )

    validated = apply_severity_engine(validated, claim_object, extracted, preflight)
    confidence = compute_confidence(
        validated,
        extracted,
        preflight,
        parse_retries=parse_retries,
        provider=provider,
    )
    audit["confidence"] = round(confidence, 3)
    audit["primary_output"] = {key: validated[key] for key in OUTPUT_FIELDS}

    anthropic_key, anthropic_model = _resolve_anthropic_config()
    escalated = False
    if should_escalate(confidence, validated, provider, bool(anthropic_key)):
        verify_prompt = build_verification_prompt(user_claim, claim_object, validated, extracted)
        try:
            verified, verify_retries = await _run_model_pass(
                "anthropic",
                anthropic_key,
                anthropic_model,
                "",
                image_entries,
                verify_prompt,
                claim_object,
                call_kind="verify",
            )
            verified = apply_severity_engine(verified, claim_object, extracted, preflight)
            validated = verified
            escalated = True
            _usage_stats["escalations"] += 1
            confidence = compute_confidence(
                validated,
                extracted,
                preflight,
                parse_retries=parse_retries + verify_retries,
                provider="anthropic",
            )
            audit["confidence"] = round(confidence, 3)
            audit["verify_output"] = {key: validated[key] for key in OUTPUT_FIELDS}
        except Exception as exc:
            audit["escalation_error"] = str(exc)

    audit["escalated"] = escalated
    if collect_audit or escalated:
        audit["final_output"] = {key: validated[key] for key in OUTPUT_FIELDS}
        _audit_trails.append(audit)

    return {**base_result, **validated}
