# ZentroMart VoiceOps capstone

The project includes a prompt-only agent, multilingual RAG, identity-gated tools,
deterministic policy calculations, stateful conversation, Gnani speech, an audio-folder
runner, and a local dashboard with saved chat history.
See [the action plan](PLAN_OF_ACTION.md).

The current retrieval model is **Google Gemini Embedding 1
(`gemini-embedding-001`, 768 dimensions)**. Vectors stay in the local index;
embedding creation/search use the Google API. Earlier E5 retrieval measurements
in the phase reports are historical and do not establish Gemini retrieval scores.
Answer generation is opt-in through `LLM_CALLS_ENABLED=true`. The final evaluation
uses the user-selected 234-request scope with shared pacing below 100 RPM.

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
| `LLM_CALLS_ENABLED` | Controls answer generation/grading; independent of embedding calls |
| `EMBEDDING_PROVIDER` | `gemini` by default; `local` explicitly selects the earlier E5 implementation |
| `EMBEDDING_API_KEY` | Optional dedicated Google key; otherwise uses the Google `LLM_API_KEY` |
| `GNANI_STT_MODEL` | Explicit STT model, `prisma-v3.0`; clips are capped at 30 seconds |

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
completed project. The final selected scope and its request audit are documented below.

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

`LLM_CALLS_ENABLED=false` disables answer generation and grading. Gemini embeddings
use the network independently; select `EMBEDDING_PROVIDER=local` explicitly for
cached E5 inference. Build and inspect retrieval with:

```bash
# Build Gemini Embedding 1 vectors for public KB/catalog chunks.
python -m src index

# Optional legacy E5 mode, selected explicitly:
EMBEDDING_PROVIDER=local python -m src index --download-model

# Gemini reuses cached vectors; uncached queries use the embedding API.
python -m src index
python -m src retrieve "UPI refund ka time kitna hai?"
python -m src retrieve "electronics return window" --delivered-on 2026-07-31
python -m src eval-retrieval --input both --k 12
```

The optional CPU embedding model is quantized `intfloat/multilingual-e5-small`, loaded
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
the saved final evaluation measures generated-answer quality separately.

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

Conversation intent extraction and automatic tool routing are now implemented in
Phase 4; the Phase 3 services also remain callable independently.
See [the implementation report](PHASE_2_3_REPORT.md) for measured results.

For a new billable run, enable LLM calls. The same runner supports both stages:

```bash
python -m src baseline --stage baseline --input both --judge
python -m src baseline --stage rag --input both --judge
python -m src ask --stage rag "How fast is a wallet refund?"
```

## Phase 4: stateful conversation

```bash
# Local interactive text chat; /quit or EOF ends the session.
python -m src chat

# Add per-turn validation, routing and tool feedback.
python -m src chat --debug

# All 23 practice conversations / 96 turns, with no Gemini calls.
python -m src eval-conversations

# With LLM_CALLS_ENABLED=true:
python -m src chat --live
python -m src eval-conversations --live
```

`ConversationAgent(SupportOperations(...), rag=optional_rag_agent)` accepts
`CustomerInput(text, session_id)` and returns the shared `AgentResponse`. The
application must use a fresh session ID for each customer conversation. The
controller holds session state in memory and `clear_session` clears its identity
state too. It handles identity prompts/retries, language changes, topic and order
switches, return/refund and delivery follow-ups, one-question clarification,
policy-grounded calculations and all documented escalation triggers. New orders
need their own verification; switching to public policy preserves the current
order's verification but discards unrelated issue context. No order mutation tool
is invented, and a ticket is confirmed only after successful creation.

Offline mode uses deterministic operational replies and a limited policy fallback;
open-policy fallback may show original English source extracts. `--live` adds
RAG for open policy questions using sanitized relevant context. It respects the
existing generation setting. Gemini was not called during Phase 4 development;
the final comparison is a separate frozen experiment.

Chat explains incomplete input with concrete examples instead of repeating a
generic prompt. Order IDs must have exactly 8 digits; malformed IDs are rejected
before lookup, and invalid verification formats do not spend an identity retry.
"New order" asks whether you mean placing an order or checking an existing order;
placing orders is explicitly unavailable in this support chat.

Use `/help` for examples, `/status` for the current mode and last result, `/reset`
for a fresh conversation, and `/quit` to exit. `--debug` prints a safe diagnostic
record after each turn: result code, decision status, recognized intent, missing
field, identity-attempt count, tool names/statuses, evidence IDs and whether an LLM
was used. It excludes customer input, proof values and private order records.
Offline mode clearly identifies Gemini as disabled. Live generation failures show
a failure code and keep the chat open for retry; they do not become an answer.
The same per-turn diagnostics are included in serialized conversation responses.

`eval-conversations` creates `runtime/conversations/RUN_ID/manifest.json`,
`results.jsonl`, `summary.json` and `report.md`. Sessions are isolated and turns
ordered; only customer messages reach the controller. Evaluator metadata stays
outside the agent, and proof-bearing messages/expected proof arguments are not
saved. The runner checks decisions and tool names/counts and reports evidence
metrics; these checks are not semantic answer accuracy. Partial outputs survive
interruptions, and persistent live service failures halt the run.

The saved local replay completed all 96 turns with matching expected decisions
and tool names/counts, across all 23 scenarios. The current 124-test suite includes
session/order isolation, language switches, privacy, clarification, escalation,
ticket failure and replay regressions. See [the Phase 4 report](PHASE_4_REPORT.md)
for artifacts and limitations.

## Local website

Start the browser workspace from the project directory:

```bash
.venv/bin/python -m src web
# Optional alternate local port:
.venv/bin/python -m src web --port 8080
```

Open [http://localhost:8000](http://localhost:8000). The server stays running until
Ctrl+C. The website uses the same `ConversationAgent` as the text console, with a
responsive solid-color theme, suggested questions, a message composer, source
references and a conversation-details panel. It shows invalid input, verification,
tool success/failure and whether generation was used. New conversation clears the
previous active verification while preserving its saved transcript. Recent chats
appear in the sidebar and can be reopened or deleted. Reload restores saved chat
history; after server restart the caller must verify again before private lookups.

The default answer mode is offline. `LLM_CALLS_ENABLED=true python -m src web --live`
enables RAG policy generation. Record or upload up to 30 seconds of English, Hindi,
or Hinglish speech; Gnani transcribes and synthesizes the reply. The browser uses
an AudioWorklet to capture WAV audio and provides reply playback. Microphone
permission is requested only when Record is clicked. No website is deployed.

The frontend is plain HTML/CSS/JavaScript in `web/`; no Node installation, frontend
build, CDN, external fonts or extra Python dependencies are required. `src/web.py`
serves assets and same-origin JSON endpoints on loopback only. It issues opaque,
HttpOnly/SameSite cookies, expires idle sessions after one hour, sanitizes displayed
proofs and response traces, and rejects cross-origin requests. Raw model output
and private dataset files are not served. Request IDs make message retries
idempotent, including failed identity checks. Sanitized chat history and voice
replies are stored in `runtime/web_history.sqlite3`, scoped to the browser's opaque
cookie. Proof values and verified identity state are not persisted. Treat this
runtime file as private local data and exclude it from releases.

This server is for local use. A later deployment can retain the frontend and
controller while adding an appropriate production server, HTTPS, authentication
and session storage for its hosting environment.

```bash
# Includes real loopback HTTP integration tests.
.venv/bin/python -m unittest discover -s tests -q
```

## Speech and audio-folder runner

Install `ffmpeg` alongside Python. Fill in `GNANI_API_KEY`, the LLM settings and
the embedding key in `.env`, then build the index with `python -m src index`.
The required interface is:

```bash
LLM_CALLS_ENABLED=true python run.py --input-dir ./test_audio --output-dir ./outputs/run1 --dataset-dir ./Dataset
# Deterministic local text replies; speech still uses the Gnani API:
python run.py --input-dir ./test_audio --output-dir ./outputs/local1 --offline
```

Supported audio: WAV, MP3, FLAC, OGG, M4A, AAC and WebM, up to 20 MB and 30 seconds
per clip. The runner normalizes audio to mono 16 kHz PCM. `--language` accepts
`en-IN`, `hi-IN`, `hi-en` or `indic-auto`. Gnani `prisma-v3.0` performs STT and
`timbre-v2.5` / `Kaveri` performs TTS. Hindi/English digit words are normalized
conservatively; ambiguous identifiers trigger a repeat request.

Files are independent conversations unless `--manifest conversation.json` is
provided. A manifest is an ordered JSON array, for example:

```json
[
  {"conversation_id":"refund-demo","turn":1,"file":"question.wav","language":"en-IN"},
  {"conversation_id":"refund-demo","turn":2,"file":"follow-up.wav","language":"en-IN"}
]
```

Paths must stay inside the input directory and turns must start at one without
gaps. Each `clip-0001/` output saves `reply.txt`, playable `reply.wav`, original
used chunks in `evidence.json`, and a sanitized `result.json` with transcript,
decision, tool events and stage timings. `manifest.json` fingerprints inputs,
source and configuration; `summary.json` distinguishes completed, needs-repeat
and failed clips. Exit 0 means every clip completed; exit 2 means partial voice
completion; exit 1 means a configuration/input failure.

Use the same command with `--resume` to reuse completed artifacts. TTS-only
failures retry synthesis from a saved text checkpoint. Partial conversations may
replay preceding audio to reconstruct state; ticket creation remains idempotent.
Changed source, configuration or inputs refuse resume. Silent/corrupt audio gets
a repeat prompt and is counted as `needs_repeat`, never a completed support turn.

Live checks used synthetic Gnani speech in English, Hindi and Hinglish. The mixed
batch completed all three valid clips and correctly requested repeats for silence
and corruption. A clean installation also completed one English clip. Human
recordings and microphone recognition have not been evaluated. The short demo is
playback of real saved agent outputs with synthetic caller speech.

## Embedding requests and final evaluation scope

Gemini Embedding 1 uses 100-item synchronous batches. The 154 public KB/catalog
chunks need two successful document-embedding requests on an empty cache.
Vectors are normalized, versioned and cached persistently. Raw query text and keys
are not written into the cache. An API failure never silently switches models.
If the second batch is rate limited, rerunning `index` reuses the first batch.
`LLM_CALLS_ENABLED=false` blocks answer generation, not the requested embedding API.

Evaluation prefetches 52 query vectors in one embedding request, then reuses those
vectors for the per-question retrievals. Automatic grading groups up to ten saved
answers per request and preserves its raw batch output. It remains a provisional
model judgement rather than human-reviewed or organizer accuracy.

With the index already built, the available scopes are:

| Scope | Answer calls | Query-embedding calls | Grading calls | New HTTP calls |
| --- | ---: | ---: | ---: | ---: |
| Smoke: 5 baseline + 5 RAG | 10 | 1 | 0 | 11 |
| Required: 52 baseline + 52 RAG | 104 | 1 | 0 | 105 |
| Required + batched grading | 104 | 1 | 12 | 117 |
| Original + ASR, with batched grading | 208 | 2 | 24 | 234 |

These are planned successful HTTP requests with a cold query cache; retries can
add calls. Batched texts and token counts are separate quantities. The user selected
the 234-call scope. A shared rolling-window limiter allows 90 quota units in 60
seconds across both workers and all retries; each embedded text counts as one unit
conservatively. HTTP 429 pauses both workers and honors server retry delays.
Provider token/day quotas can still cause rate limits.
The implementation is in `scripts/final_evaluation.py` (`--limit 5`, `--judge`,
`--asr`), and never enables the optional variants by default.

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
  allowlist. The verification controller must supply known proof/API-key
  values when serializing; this function does not infer every secret from prose.
- `src/evaluation/`: labels remain evaluator-only. The agent callback receives
  `CustomerInput(text, session_id)`, without reference answers or gold evidence.
  The question runner saves actual responses, decision matches, and per-question
  evidence metrics; optional separate semantic grading scores factual checks.
  Conversation cases replay in order through the Phase 4 controller.

`run_questions(cases, respond, output_dir)` takes the real Phase 1 agent callback.
It creates a new output directory, refuses to overwrite past results, and records
individual failures while continuing the run. Keep experiment metadata/model
settings with each frozen baseline/RAG comparison.

## Current completion status

- Offline foundation and supplied dataset validation: complete.
- Automated foundation/integration-contract checks: available in `tests/`.
- Phase 1 agent and reproducible evaluation code: implemented and tested locally.
- Live smoke reports: see `runtime/smoke/*/checks.json` for timestamped results.
- Local multilingual retrieval, source provenance and Phase 3 services: implemented.
- Phase 4 conversation controller and ordered replay: implemented.
- 124 local tests pass, including speech, history, shared RPM pacing, batched embeddings, grading contracts and HTTP integration tests.
- All 23 practice conversations / 96 turns pass decision and tool-name/count checks; semantic accuracy remains unscored.
- Gnani speech, batch processing, dashboard playback and saved chat history are implemented.
- Fresh dependency installation and execution with an external dataset passed.
- Final Gemini generation/grading uses the selected 234-call scope. Saved results
  and limitations accompany the release; model grades remain provisional.

The configured hosted model is `gemini-3.5-flash-lite`: it responded successfully while
`gemini-3.8-flash` reported temporary overload. Use the same available model for
the Step 1/Step 2 comparison. Keys were moved from the example into ignored `.env`.

## Integration references

- [Gemini embedding guide](https://ai.google.dev/gemini-api/docs/embeddings) and [embedding API](https://ai.google.dev/api/embeddings) for `gemini-embedding-001`, batching, task types and dimensionality.
- [Official OpenAI Chat Completions HTTP contract](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create) for `model`, `messages`, and text response shape. This adapter supports that contract; other provider protocols need a separate adapter.
- [Official Gnani SDK](https://github.com/Gnani-AI-Mintlify/Gnani-Vachana/blob/main/README.md) and [published package](https://pypi.org/project/gnani-vachana/).
- [python-dotenv](https://pypi.org/project/python-dotenv/) for local `.env` support.

## Frozen final results and local deliverables

| Input | Prompt-only provisional pass | RAG provisional pass |
| --- | ---: | ---: |
| Original text | 7/52 (13.5%) | 24/52 (46.2%) |
| ASR-like text | 8/52 (15.4%) | 30/52 (57.7%) |

The answer-quality target was not met. RAG had 11 original and 10 ASR contract
failures; all remain in the denominator. These are provisional model grades,
without human review. Gemini candidate recall@12 was 92.2% and enriched-context
recall 100% for both variants. Routing checks are a separate measurement.

The selected evaluation completed 234 successful HTTP requests in 248 attempts,
including 14 rate-limit retries. The independently verified peak was 70 quota
units per rolling minute, under the shared 90-unit cap and user limit of 100 RPM.
No extra holdout or conversation-semantic calls were made. Post-run analysis used
cached vectors and made zero provider calls. Inspect the frozen evidence without
rerunning the billable experiment in `sample_outputs/evaluation/` and
`sample_outputs/analysis/`.

See [the final phase report](PHASE_5_7_REPORT.md), [four-page report](report.pdf),
[synthetic playback demo](demo.mp4), and [actual sample outputs](sample_outputs/).
`zentromart_voiceops.zip` is the local submission package. Public repository
synchronization and deployment have not been performed. Human-recorded speech,
human semantic review and unseen-test performance remain unverified.
