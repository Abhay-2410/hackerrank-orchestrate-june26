# Multi-Modal Evidence Review

Python pipeline that verifies damage claims using a layered **signature stack**: conversation extraction, image pre-flight checks, evidence requirement routing, a primary vision LLM, severity calibration, and optional Anthropic verification for low-confidence cases.

Supports **Anthropic Claude**, **Ollama cloud**, and **Google Gemini** — provider is chosen automatically from `.env`.

## Prerequisites

- Python 3.9+
- At least one API key (see Setup)

## Setup

```bash
pip install -r code/requirements.txt
```

Copy `.env.example` to `.env` in the **repository root**:

```env
# Recommended for final submission (best accuracy — auto-selected when set)
ANTHROPIC_API_KEY=sk-ant-api03-your-key-here

# Free fallback (primary model)
OLLAMA_API_KEY=your-ollama-key
OLLAMA_MODEL=gemma4:31b
OLLAMA_HOST=https://ollama.com

# Optional: escalation threshold (default 0.62)
ESCALATION_THRESHOLD=0.62
```

**Provider priority:** `ANTHROPIC_API_KEY` → `OLLAMA_API_KEY` → `GEMINI_API_KEY`

When primary is Ollama/Gemini and `ANTHROPIC_API_KEY` is set, uncertain claims are automatically verified with Claude.

## Run

From the repository root:

```bash
# Step 1 — evaluate on labeled sample set (20 claims)
python code/evaluation/main.py

# Step 1b — offline refine output.csv with zero API calls (recommended after any model run)
python code/refine_offline.py

# Step 2 — generate final predictions (44 test claims)
python code/main.py
```

Or use the refinement script (eval + main + package):

```powershell
.\scripts\refine.ps1
```

| Script | Input | Output |
|---|---|---|
| `code/evaluation/main.py` | `dataset/sample_claims.csv` | Terminal accuracy + `code/evaluation/evaluation_report.md` + `code/evaluation/audit/*.json` |
| `code/main.py` | `dataset/claims.csv` | `output.csv` (repo root) + `code/evaluation/run_stats.json` |

## Project layout

```
code/
├── main.py                      # Batch prediction entry point
├── claim_processor.py           # Signature stack orchestration
├── conversation_extractor.py    # Structured claim understanding from chat
├── image_preflight.py           # Blur/glare/resolution CV pre-checks
├── evidence_checker.py          # Evidence requirement routing by applies_to
├── severity_engine.py           # Post-model consistency + severity calibration
├── output_calibrator.py         # Zero-API refinement rules (issue/severity/evidence)
├── confidence.py                # Confidence scoring + escalation gate
├── prompt_builder.py            # Structured prompt with routed context
├── history_lookup.py            # User history lookup by user_id
├── refine_offline.py            # Re-process output.csv with 0 API calls
├── requirements.txt
├── README.md
└── evaluation/
    ├── main.py                  # Sample-set evaluation + error analysis
    ├── evaluation_report.md
    ├── run_stats.json           # Latest test-set run metrics (written by main.py)
    └── audit/                   # Per-claim audit JSON (written during evaluation)
```

## Signature stack flow

```
claims.csv row
      │
      ▼
conversation_extractor ──► claimed issue/part, issue families, injection flags
      │
      ▼
image_preflight ──► blur, glare, resolution, duplicate detection
      │
      ▼
evidence_checker (router) ◄── evidence_requirements.csv (matched by applies_to)
      │
      ▼
history_lookup ◄── user_history.csv
      │
      ▼
prompt_builder ──► structured analysis prompt
      │
      ▼
claim_processor ──► primary vision LLM (1 call per claim)
      │
      ▼
severity_engine ──► cross-field consistency + severity calibration
      │
      ▼
confidence gate ──► escalate if low confidence?
      │                    │
      │                    ▼ (optional)
      │              Anthropic verify call
      ▼
output.csv row (14 columns)
```

### Concurrency

Both entry points process up to **3 claims in parallel** (`asyncio.Semaphore(3)`).

## Output schema

`output.csv` columns (exact order): `user_id`, `image_paths`, `user_claim`, `claim_object`, `evidence_standard_met`, `evidence_standard_met_reason`, `risk_flags`, `issue_type`, `object_part`, `claim_status`, `claim_status_justification`, `supporting_image_ids`, `valid_image`, `severity`

- `supporting_image_ids`: `img_1;img_2` or `none`
- `evidence_standard_met` / `valid_image`: lowercase `true` / `false`

## Submission checklist

| Deliverable | Location |
|---|---|
| `code.zip` | `.\scripts\package.ps1` (excludes `__pycache__`) |
| `output.csv` | Repo root |
| `chat_transcript` | `%USERPROFILE%\hackerrank_orchestrate\log.txt` |

```powershell
.\scripts\package.ps1
```

## Security

- Never commit `.env` or hardcode API keys
- Keys loaded from environment variables only
