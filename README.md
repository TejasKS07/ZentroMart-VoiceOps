# ZentroMart VoiceOps capstone

Phases 0–3 provide configuration, the prompt-only agent, local multilingual RAG,
identity-gated tools and deterministic policy calculations. Conversation routing,
interactive voice and the submission audio-folder runner are later phases.
See [the action plan](PLAN_OF_ACTION.md).

## Setup

Use Python 3.12 or newer (the pinned NumPy version requires 3.12). Dependencies were checked with
Python 3.12. Keep the organizer dataset in `Dataset/`, or set `DATASET_DIR` to its
extracted root. The dataset is an external input and must not be included in the
submission ZIP.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
# Only for a new checkout; preserve an existing .env with your credentials.
test -f .env || cp .env.example .env
```

With uv, the equivalent installation is:

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

Edit `.env` locally. Existing environment variables override the file. Relative
configuration paths resolve against this project directory. Nothing imports API
clients or makes network calls as a side effect of loading the application.

| Setting | Purpose |
| --- | --- |
| `DATASET_DIR` | Dataset root containing `kb/`, `data/`, `practice/`, and supplied validator/tools |
| `RUNTIME_DIR` | Generated outputs, outside the dataset; defaults to `runtime/` |
| `LLM_ENDPOINT` | Complete HTTPS text Chat Completions URL, including `/chat/completions`; localhost HTTP is allowed |
| `LLM_MODEL` | Exact model ID from your chosen provider; no default model is assumed |
| `LLM_API_KEY` | Provider bearer key; may be blank for a local unauthenticated endpoint |
| `GNANI_API_KEY` | Participant's own Gnani key |
| `GNANI_TTS_MODEL` / `GNANI_TTS_VOICE` | Explicit speech settings; initial candidates `timbre-v2.5` / `Kaveri` |
| `GNANI_LANGUAGE` | Initial speech locale, `en-IN` |
| `API_TIMEOUT_SECONDS` | Per-request timeout, default 45 seconds |
| `LLM_TEMPERATURE` | Optional 0–2; blank leaves the provider default |
| `LLM_REASONING_EFFORT` | Optional provider-supported effort; blank leaves its default |
| `LLM_CALLS_ENABLED` | Defaults to `false`; keep disabled until the full project is complete |

All policy calculations use **20 October 2026**; the application does not take
"today" from the computer clock. Real timestamps may still identify test runs.

`.env` is ignored and contains real credentials locally; `.env.example` must keep
key values empty. To select another environment file explicitly, place the global
option before the command: `python -m src --env-file /path/to/settings.env check`.

## Offline checks

Run commands from the project directory using the activated virtual environment:

```bash
python -m src check --validate
python -m src eval-inventory
python -m unittest discover -s tests -v
```

An external dataset can be selected with:

```bash
python -m src --dataset-dir /path/to/extracted/dataset check --validate
```

`check --validate` runs the organizer's full validator, then checks data loading,
primary keys, public document parsing, practice input structure, and dependency
versions. It prints counts/configuration readiness without private records or
keys. `offline_passed` does not establish API connectivity. `eval-inventory`
reports practice counts only; it does not call an agent or invent scores.

## Live connectivity checks

Configure credentials, the model/endpoint, and `LLM_CALLS_ENABLED=true` first. These commands make real API
calls and may incur provider charges:

```bash
python -m src smoke --component llm
python -m src smoke --component gnani
python -m src smoke --component gnani --audio /path/to/your-own-recording.wav
```

The LLM check sends a short neutral text prompt to the configured endpoint. The
Gnani check synthesizes a short test sentence into a valid WAV, then transcribes
the supplied recording, or the synthesized WAV if no recording is specified.
Loopback verifies connectivity only; it is not real-caller ASR evaluation.
Use your own short recording for a meaningful microphone check.

Generated smoke audio and a `checks.json` report are saved in `runtime/smoke/<unique-run>/`. Transcripts and
provider error bodies are not printed or saved. Missing settings produce
`pending` and exit code 2; failed checks also exit 2 with status `failed`. A passed
check requires a successful real response. The offline tests use doubles and do
not count as live checks.

The Gnani SDK is pinned to `gnani-vachana==0.9.0`; constructor and method signatures
were inspected locally. Model/voice/account availability still requires the live
check. No non-Gnani speech service is used.

## Phase 1: prompt-only agent and baseline

Ask one question in a fresh support conversation:

```bash
python -m src ask "How long does a refund take?"
```

The system prompt establishes the support role, AI disclosure, language matching,
limits and output contract. It contains no company policy facts or example
answers. The agent receives only that prompt and the customer's message: no KB,
catalog rows, order details, tools, labels or conversation history. It can
acknowledge missing information; a low baseline score is valid evidence for RAG.

Baseline generation and optional grading use the configured external LLM and
therefore send synthetic practice content to that provider. Generation sends only
questions; grading additionally sends the reference answer, required/forbidden
facts, expected language and candidate reply. It never sends the private customer,
order or support-log tables. The user has authorized these calls for the final
completed project; keep them deferred during development without asking again.

```bash
# Two-question integration check; does not count as a full baseline.
python -m src baseline --limit 2 --judge

# Full 52-question baseline, followed by separate semantic grading.
python -m src baseline --judge

# Original and ASR-like text as two independent experiments.
python -m src baseline --input both --judge

# Grade a completed generation run without regenerating its answers.
python -m src grade runtime/baseline/RUN_ID
```

Each run creates a unique `runtime/baseline/RUN_ID/` directory containing:

- `manifest.json`: prompt text/version/hash, requested model, explicit generation
  parameters, source and practice-data hashes, runtime/dependency versions,
  selected question IDs and whether this is a full 52-question run.
- `results.jsonl`: actual unedited replies, raw model text, resolved model,
  finish reason, token usage, latency, decision match and evidence metrics.
- `summary.json`: completion/failure counts and decision match rate. This rate
  is not factual answer accuracy.
- `judgements.jsonl`, `judge_manifest.json`, `grading_summary.json` when grading
  is requested. Semantic checks accept paraphrases and equivalent multilingual
  facts. A pass requires an accepted decision, required facts, no forbidden
  claims, correct reply language and first-turn AI disclosure. Model-judged
  scores are provisional and need human spot-checks.
- `report.md`: readable results and the comparison procedure for Step 2.

The entire answering pass is saved before grading begins, keeping reference
answers out of generation. Malformed/truncated model responses count as contract
failures and preserve their original output; they are not rewritten or repeatedly
regenerated until acceptable. Provider outages stay distinct from factual errors.
Transient network/HTTP failures have at most three attempts; persistent failures
halt a run after three affected questions. Authentication/configuration HTTP
errors halt immediately. Partial results remain available after interruption.
Grading refuses to overwrite an existing judgment file.

Keep the same model, generation parameters, case selection, input variant and
judge procedure when comparing Step 2. Provider defaults apply for unspecified
generation parameters; the manifest records this limitation and per-response
model IDs. AI disclosure and factual correctness are measured, not inserted into
or corrected in the model's saved reply.

## Phases 2 and 3: local retrieval and verified tools

Development currently makes **no Gemini calls**. The user authorized Gemini
generation/grading once the entire project is complete; `LLM_CALLS_ENABLED=false`
enforces that deferral, including smoke, ask, baseline and judge calls.
The following commands run locally:

```bash
# First installation only: download public embedding weights, then build index.
python -m src index --download-model

# Subsequent builds/searches use cached weights with network downloads disabled.
python -m src index
python -m src retrieve "UPI refund ka time kitna hai?"
python -m src retrieve "electronics return window" --delivered-on 2026-07-31
python -m src eval-retrieval --input both --k 12
```

The CPU embedding model is quantized `intfloat/multilingual-e5-small`, loaded
through FastEmbed/ONNX Runtime with the required query/passage prefixes. The
public model/tokenizer download is approximately 140 MB; cached inference stays
local. Index metadata checks source, weight and tokenizer fingerprints and
rejects stale indexes. The index has 154 chunks: 84 KB sections and 70 catalog
rows. Private orders, customers, support logs and practice answer keys are excluded.

Search combines semantic similarity, lexical overlap and document authority.
It resolves public product names/SKUs, labels ambiguous spoken SKU candidates,
and retains current successor rules when a historical delivery date makes an
older policy applicable. The context includes top candidates, document neighbors,
explicit cross-references, KB rules and agent guides. Candidates and enriched
context are reported separately. These contextual additions improve coverage at
the cost of extra input tokens during the final live run.

`RagAgent` records actually cited source chunks and per-claim evidence IDs,
rejects unknown citations and unsupported digit-form numeric claims, and permits explicit
abstention. These structural/numeric checks do not prove semantic correctness;
actual generated-answer quality will be tested after project completion.

The Phase 3 API is `SupportOperations(dataset_dir, runtime_dir)`:

```python
from pathlib import Path
from src.tools.operations import SupportOperations

ops = SupportOperations(Path("Dataset"), Path("runtime/tools"))
session = "current-call"
pending = ops.gateway.begin(session, order_id)  # customer-provided order ID
verified = ops.gateway.verify(session, caller_proof)  # one caller-provided proof
if verified.verified:
    result = ops.refund_status(session, "return")
```

The gateway uses the supplied `lookup_order` and `create_ticket` implementations
without changing their source. It permits two identity attempts, keeps attempts
when switching back to an order, scopes verified facts by session/order, and
removes expected proof/contact fields before model context. Backend failures are
not charged as wrong identity attempts. Use `gateway.sanitize(session, value)` at
persistence boundaries; session proof values stay in private memory for redaction.

`gateway.handover(session, issue_key=..., issue=..., intent=..., checked=...,
reason=...)` builds a sanitized summary and returns a tool event. Claim handover
success only for `status="succeeded"` and `details["created"] is True`. A local
SQLite idempotency ledger suppresses repeated successful tickets across restarts.
Generated tickets and the ledger are under the runtime directory, never Dataset.

Public product/pincode lookups include source rows. Verified return/refund calls
use selected item lines and quantities. `PolicyCalculator` reads current category
windows, historic/festive exceptions, fees, settlement ranges and escalation
thresholds from the actual KB; holidays come from the CSV. It distinguishes
calendar days, Mon–Sat business days, refund initiation and settlement. Date-only
wallet initiation cannot establish an exact two-hour deadline. A missing grocery
slot or current time requires clarification. Quotes use Decimal and half-up rupee
rounding, coupon allocation/clawback, fee waivers and whole-order fee rules.
Canonical quote reasons are `change_of_mind`, `damaged`, `defective`, and `wrong`;
dataset labels such as `quality` must not be blindly treated as approved defects.
Quotes and window checks are estimates/conditions, never approval or execution.

Conversation intent extraction and automatic tool routing remain Phase 4 work;
the completed Phase 3 services are callable and tested independently.
See [the implementation report](PHASE_2_3_REPORT.md) for measured results.

After the full project is complete, enable LLM calls for the already-authorized
live runs. The same runner supports both stages:

```bash
python -m src baseline --stage baseline --input both --judge
python -m src baseline --stage rag --input both --judge
python -m src ask --stage rag "How fast is a wallet refund?"
```

## Contracts and boundaries

- `src/config.py`: environment settings and the fixed dataset reference date.
- `src/dataset.py`: UTF-8 CSV reading preserves every value as text, including
  leading zeroes and empty cells. Public knowledge contains catalog, pincodes,
  holidays, and policy documents; no customer/order/support-log tables or practice
  answers are exposed in that object. Inventory reads private tables only for
  counts/key checks. Private order access uses the supplied verified tool through
  the Phase 3 gateway.
- `src/schemas.py`: validated decisions, used evidence with original chunk text
  and metadata, candidate retrieval IDs, language, timings, and tool-event output.
  Used evidence and retrieved candidates are distinct.
- `src/redaction.py`: masks sensitive keys and explicitly supplied secret values,
  including grouped/native-script numeric proof. Tool events persist only a small
  allowlist. The future verification controller must supply known proof/API-key
  values when serializing; this function does not infer every secret from prose.
- `src/evaluation/`: labels remain evaluator-only. The agent callback receives
  `CustomerInput(text, session_id)`, without reference answers or gold evidence.
  The question runner saves actual responses, decision matches, and per-question
  evidence metrics; optional separate semantic grading scores factual checks.
  Conversation cases load in order, but replay awaits Phase 4.

`run_questions(cases, respond, output_dir)` takes the real Phase 1 agent callback.
It creates a new output directory, refuses to overwrite past results, and records
individual failures while continuing the run. Keep experiment metadata/model
settings with the baseline when that phase is implemented.

## Current completion status

- Offline foundation and supplied dataset validation: complete.
- Automated foundation/integration-contract checks: available in `tests/`.
- Phase 1 agent and reproducible evaluation code: implemented and tested locally.
- Live smoke reports: see `runtime/smoke/*/checks.json` for timestamped results.
- Local multilingual retrieval, source provenance and Phase 3 services: implemented.
- 56 offline tests pass, including actual supplied-tool integration tests.
- Gemini generation/grading is authorized for final project evaluation and is
  deliberately deferred until then. No generated-answer accuracy is claimed.

The configured hosted model is `gemini-3.5-flash-lite`: it responded successfully while
`gemini-3.8-flash` reported temporary overload. Use the same available model for
the Step 1/Step 2 comparison. Keys were moved from the example into ignored `.env`.

## Integration references

- [Official OpenAI Chat Completions HTTP contract](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create) for `model`, `messages`, and text response shape. This adapter supports that contract; other provider protocols need a separate adapter.
- [Official Gnani SDK](https://github.com/Gnani-AI-Mintlify/Gnani-Vachana/blob/main/README.md) and [published package](https://pypi.org/project/gnani-vachana/).
- [python-dotenv](https://pypi.org/project/python-dotenv/) for local `.env` support.
