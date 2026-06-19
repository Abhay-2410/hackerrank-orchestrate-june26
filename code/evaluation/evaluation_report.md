# Evaluation Report

Total sample claims evaluated: 20

## Signature stack architecture

Pipeline: conversation extraction → image pre-flight CV → evidence requirement routing →
primary vision LLM → severity/consistency engine → confidence gate → optional Anthropic verification.

- Primary model calls: 20
- Verification escalations: 0
- Verify model calls: 0

## Per-field accuracy (current strategy)

| Field | Accuracy |
|---|---:|
| claim_status | 85.0% |
| evidence_standard_met | 90.0% |
| issue_type | 75.0% |
| severity | 75.0% |
| valid_image | 95.0% |

## claim_status confusion matrix

| expected \ predicted | contradicted | not_enough_information | supported |
|---|---|---|---|
| contradicted | 2 | 1 | 2 |
| not_enough_information | 0 | 3 | 0 |
| supported | 0 | 0 | 12 |

## severity confusion matrix

| expected \ predicted | high | low | medium | none | unknown |
|---|---|---|---|---|---|
| high | 1 | 0 | 0 | 0 | 0 |
| low | 0 | 2 | 0 | 1 | 0 |
| medium | 2 | 0 | 9 | 0 | 0 |
| none | 0 | 1 | 1 | 0 | 0 |
| unknown | 0 | 0 | 0 | 0 | 3 |

## severity accuracy by claim_object

- car: 100.0%
- laptop: 50.0%
- package: 66.7%

## Strategy comparison

| Strategy | claim_status | evidence_standard_met | issue_type | severity | valid_image |
|---|---:|---:|---:|---:|---:|
| Gemini 2.5-flash-lite (free tier) | 50.0% | 60.0% | 75.0% | 60.0% | 80.0% |
| Ollama gemma4:31b (free tier) | pending | pending | pending | pending | pending |
| Anthropic claude-sonnet-4-6 (recommended final) | 85.0% | 90.0% | 75.0% | 75.0% | 95.0% |

**Final strategy for output.csv:** anthropic / claude-sonnet-4-6

## Operational summary (sample set)

- Provider / model: anthropic / claude-sonnet-4-6
- Approximate API calls (sample): 20
- Approximate input tokens: 62,016
- Approximate output tokens: 7,392
- Images processed: 29
- Approximate cost (Sonnet 4.6 @ $3.0/MTok in, $15.0/MTok out): $0.2969
- Sample evaluation runtime: 67s (3.4s per claim)
- Estimated full test runtime: 148s at same concurrency
- Estimated full test cost: ~$1.01 for 44 test claims at Sonnet 4.6 pricing

## Test set run (latest — from code/main.py)

- Timestamp (UTC): 2026-06-19T15:13:17.407023+00:00
- Provider / model: anthropic / claude-sonnet-4-6
- Test claims processed: 44
- Failed claims: 0
- API calls: 44
- Input tokens: 139,286
- Output tokens: 17,623
- Images processed: 86
- Runtime: 174s (3.96s per claim)
- Output file: `C:\Users\Admin\hackerrank-orchestrate-june26\output.csv`

## Rate limits, TPM/RPM, and batching

- Concurrency: 3 parallel claims (`asyncio.Semaphore(3)`)
- Retry: up to 3 API retries on 429/503; JSON parse retry once per claim
- Main.py auto-retries failed claims up to 2 passes
- Delay: 0.5s between successful Ollama/Gemini calls
- Images re-encoded to JPEG via Pillow before upload
- Signature stack: extract → pre-flight → route evidence → primary VLM → severity engine → escalate if low confidence
- Escalation: uncertain claims verified with Anthropic when ANTHROPIC_API_KEY is set and primary is not Anthropic
- Audit JSON written to `code/evaluation/audit/` during sample evaluation

- Anthropic Sonnet tier limits vary by account; at 3 concurrent claims ~54 RPM
- Estimated peak TPM ~185,721 (3 parallel × ~3,470 tok/claim)
- Backoff: exponential retry on 429/529 (up to 3 attempts)

## Notes

- Top failure patterns on sample set:
- user_002 (car) — evidence_standard_met: expected=false, got=True
- user_005 (car) — issue_type: expected=scratch, got=dent
- user_009 (laptop) — severity: expected=medium, got=high
- user_018 (laptop) — issue_type: expected=crack, got=glass_shatter, severity: expected=medium, got=high
- user_020 (laptop) — claim_status: expected=contradicted, got=supported, issue_type: expected=none, got=scratch, severity: expected=none, got=low
- Lowest accuracy on 'issue_type' (75.0%); severity engine and Anthropic escalation target this gap.
- Refinement: add ANTHROPIC_API_KEY to .env for primary or escalation verification, then run `scripts/refine.ps1` (or eval + main manually).
