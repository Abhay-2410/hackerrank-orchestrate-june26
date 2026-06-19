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
| evidence_standard_met | 70.0% |
| issue_type | 85.0% |
| severity | 65.0% |
| valid_image | 75.0% |

## claim_status confusion matrix

| expected \ predicted | contradicted | not_enough_information | supported |
|---|---|---|---|
| contradicted | 4 | 0 | 1 |
| not_enough_information | 0 | 2 | 0 |
| supported | 1 | 1 | 11 |

## severity confusion matrix

| expected \ predicted | high | low | medium | none | unknown |
|---|---|---|---|---|---|
| high | 1 | 0 | 0 | 0 | 0 |
| low | 0 | 0 | 1 | 2 | 1 |
| medium | 2 | 0 | 9 | 0 | 0 |
| none | 0 | 0 | 1 | 1 | 0 |
| unknown | 0 | 0 | 0 | 0 | 2 |

## severity accuracy by claim_object

- car: 62.5%
- laptop: 66.7%
- package: 66.7%

## Strategy comparison

| Strategy | claim_status | evidence_standard_met | issue_type | severity | valid_image |
|---|---:|---:|---:|---:|---:|
| Gemini 2.5-flash-lite (free tier) | 50.0% | 60.0% | 75.0% | 60.0% | 80.0% |
| Ollama gemma4:31b (free tier) | 85.0% | 70.0% | 85.0% | 65.0% | 75.0% |
| Anthropic claude-sonnet-4-6 (recommended final) | pending | pending | pending | pending | pending |

**Final strategy for output.csv:** ollama / gemma4:31b

## Operational summary (sample set)

- Provider / model: ollama / gemma4:31b
- Approximate API calls (sample): 20
- Approximate input tokens: 44,742
- Approximate output tokens: 5,059
- Images processed: 29
- Approximate cost (Ollama/Gemini free tier (assumed $0)): $0.0000
- Sample evaluation runtime: 91s (4.6s per claim)
- Estimated full test runtime: 201s at same concurrency
- Estimated full test cost: $0 (Ollama cloud free tier assumed)

## Test set run (latest — from code/main.py)

- Timestamp (UTC): 2026-06-19T12:52:46.296368+00:00
- Provider / model: ollama / gemma4:31b
- Test claims processed: 44
- Failed claims: 0
- API calls: 44
- Input tokens: 103,491
- Output tokens: 11,753
- Images processed: 82
- Runtime: 266s (6.06s per claim)
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

- Ollama cloud: ~13 claims/min at observed latency; 0.5s delay between calls
- Estimated throughput ~98,165 TPM with concurrency 3
- Backoff: exponential retry on 429/503

## Notes

- Top failure patterns on sample set:
- user_001 (car) — severity: expected=medium, got=high
- user_002 (car) — claim_status: expected=supported, got=contradicted, evidence_standard_met: expected=true, got=False, issue_type: expected=scratch, got=unknown, severity: expected=low, got=unknown, valid_image: expected=true, got=False
- user_005 (car) — evidence_standard_met: expected=true, got=False, issue_type: expected=scratch, got=none, severity: expected=low, got=none
- user_008 (car) — evidence_standard_met: expected=true, got=False, valid_image: expected=false, got=True
- user_010 (laptop) — claim_status: expected=supported, got=not_enough_information, evidence_standard_met: expected=true, got=False, severity: expected=medium, got=high, valid_image: expected=true, got=False
- Lowest accuracy on 'severity' (65.0%); severity engine and Anthropic escalation target this gap.
- Refinement: add ANTHROPIC_API_KEY to .env for primary or escalation verification, then run `scripts/refine.ps1` (or eval + main manually).
