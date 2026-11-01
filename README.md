# ZentroMart VoiceOps

A multilingual customer-support application for ZentroMart's fictional retail
operations. It answers from company documents, handles verified order lookups and
human handovers, and supports text and Gnani speech through a local dashboard,
console, and audio-folder runner.

The configured language model is **Gemma 4 31B (`gemma-4-31b-it`)**. Retrieval uses
**Gemini Embedding 1 (`gemini-embedding-001`, 768 dimensions)** with a local vector
index. Gnani provides all speech recognition and synthesis.

Repository: [TejasKS07/ZentroMart-VoiceOps](https://github.com/TejasKS07/ZentroMart-VoiceOps).
The prepared release is local; its latest changes have not been pushed or deployed.

## Features

- Document-grounded answers with original source chunks and section IDs.
- English, Hindi, and Hinglish conversations, including follow-ups and topic changes.
- Caller verification, isolated order context, and idempotent ticket creation.
- Policy-derived return windows, refund timelines, business-day calculations, and amount estimates.
- Text chat, audio recording/upload, spoken replies, diagnostics, and saved chat history.
- Audio-folder processing with ordered conversation manifests and resumable output.
- Preserved evaluation results, raw model traces, and explicit failure reporting.

The same conversation controller serves the console, dashboard, and voice runner.
Public questions use retrieval or structured policy lookups; private order details
require verification. Audio passes through Gnani STT, the controller, and Gnani TTS.

## Requirements

- Python 3.12 or newer; the pinned dependencies were checked with Python 3.12.
- `ffmpeg` for decoding audio.
- The supplied dataset, kept outside the submission package.
- A Google API key for the configured hosted model and embeddings, plus a Gnani key
  for speech. Other text Chat Completions-compatible providers are configurable.

Provider account terms, quotas, and charges apply separately to language,
embedding, and speech requests. Local text mode does not use a language-model API;
Gnani speech and uncached Gemini embeddings still require their respective APIs.

## Installation

Run commands from the project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
# Preserve an existing configuration and its credentials.
test -f .env || cp .env.example .env
```

Alternatively, install with uv:

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
source .venv/bin/activate
```

Extract the dataset into `Dataset/`, or set `DATASET_DIR` in `.env` to its root.
That directory must contain `kb/`, `data/`, `practice/`, the validator, and supplied
tools. Add credentials only to local `.env`; keep the example's key fields blank.

Validate the configuration and dataset, then build the retrieval index:

```bash
python -m src check --validate
# Uses the embedding API unless all document vectors are cached.
python -m src index
```

The dataset validator and readiness checks do not call the model or establish API
connectivity. An external dataset can also be selected per command:

```bash
python -m src --dataset-dir /path/to/dataset check --validate
```

## Dashboard

Start the live workspace:

```bash
LLM_CALLS_ENABLED=true python -m src web --live
```

Open [http://localhost:8000](http://localhost:8000). Stop the server with Ctrl+C.
For local rules and tools without language-model generation, omit `--live`:

```bash
python -m src web
# Optional alternate port:
python -m src web --port 8080
```

The responsive dashboard provides a message composer, suggested questions,
record/upload controls, reply playback, sources, and conversation details. Status
shows missing information, verification, tool outcomes, and the configured model.
Microphone permission is requested only when recording starts.

Recent chats can be reopened or deleted. Starting a new conversation clears active
verification while preserving the old transcript. History is stored in the local,
browser-scoped `runtime/web_history.sqlite3` database. Verified identity is not
persisted, so private order access requires verification again after a restart.

The server binds to loopback and serves only fixed frontend assets and same-origin
endpoints. It uses opaque HttpOnly/SameSite cookies, rejects cross-origin requests,
and keeps private dataset files and raw model output out of browser responses.
Deployment requires a suitable production server, HTTPS, authentication, and
hosting-specific session storage.

## Console and retrieval

```bash
# Stateful local support with per-turn diagnostics:
python -m src chat --debug

# Stateful support with document-grounded language-model generation:
LLM_CALLS_ENABLED=true python -m src chat --live

# Independent prompt-only or grounded questions:
LLM_CALLS_ENABLED=true python -m src ask "How long does a refund take?"
LLM_CALLS_ENABLED=true python -m src ask --stage rag "How fast is a wallet refund?"

# Inspect retrieval without generating an answer:
python -m src retrieve "UPI refund ka time kitna hai?"
python -m src retrieve "electronics return window" --delivered-on 2026-07-31
```

Console commands include `/help`, `/status`, `/reset`, and `/quit`. Debug output
shows routing, missing fields, identity attempts, tool status, and evidence IDs;
it excludes proof values and private records. Malformed order IDs receive specific
feedback before lookup. Ambiguous requests receive one clarification question.

## Audio-folder processing

The submission interface accepts an input folder and saves each support reply:

```bash
LLM_CALLS_ENABLED=true python run.py --input-dir ./test_audio --output-dir ./outputs/run1 --dataset-dir ./Dataset
```

To use deterministic local text replies while retaining Gnani STT/TTS:

```bash
python run.py --input-dir ./test_audio --output-dir ./outputs/local1 --offline
```

Supported formats are WAV, MP3, FLAC, OGG, M4A, AAC, and WebM, up to **20 MB and
30 seconds per clip**. Audio is normalized to mono 16 kHz PCM. `--language` accepts
`en-IN`, `hi-IN`, `hi-en`, or `indic-auto`. Spoken identifiers are normalized
conservatively; ambiguous digits trigger a repeat request.

Files are independent conversations by default. Use `--manifest conversation.json`
for ordered turns:

```json
[
  {"conversation_id":"refund-demo","turn":1,"file":"question.wav","language":"en-IN"},
  {"conversation_id":"refund-demo","turn":2,"file":"follow-up.wav","language":"en-IN"}
]
```

Manifest paths must stay inside the input directory. Turns must start at one and
be consecutive. Input and output directories must be separate; output must not
be inside the dataset.

Each `clip-0001/` directory contains the available artifacts:

| File | Contents |
| --- | --- |
| `reply.txt` | Actual assembled agent reply |
| `reply.wav` | Gnani-generated speech |
| `evidence.json` | Original used chunks and their identifiers |
| `result.json` | Sanitized transcript, decision, diagnostics, tool events, timings, and error status |
| `text_checkpoint.json` | Saved text state for synthesis recovery |

The output root contains an input/configuration fingerprint manifest and a summary.
Exit code **0** means every clip completed, **2** means partial voice completion,
and **1** means a configuration or input failure. Silence, corruption, and audible
failure explanations are not counted as successful support answers.

Repeat the same command with `--resume` to reuse completed output. Synthesis-only
failures retry saved text. Partial conversations may replay earlier audio to
reconstruct state; ticket creation remains idempotent. Changed inputs, source,
or configuration refuse resume.

## Configuration

Environment variables override `.env`. Relative paths resolve against the project
root. To use another environment file, place the global option before the command:
`python -m src --env-file /path/to/settings.env check`.

| Setting | Purpose / example configuration |
| --- | --- |
| `DATASET_DIR` | Supplied dataset root; `./Dataset` |
| `RUNTIME_DIR` | Generated state outside the dataset; `./runtime` |
| `LLM_MODEL` | Hosted model ID; `gemma-4-31b-it` |
| `LLM_ENDPOINT` | Text Chat Completions URL; Google-hosted Gemma 4 is routed to its native `generateContent` API |
| `LLM_API_KEY` | Language-model provider key; may be blank for a local unauthenticated endpoint |
| `LLM_CALLS_ENABLED` | Enables generation/grading; `false` by default |
| `LLM_TEMPERATURE` | `0.2` in the example; blank uses the provider default |
| `LLM_REASONING_EFFORT` | `minimal` in the example; hosted Gemma 4 also supports `high` |
| `API_TIMEOUT_SECONDS` | `90` in the example |
| `EMBEDDING_PROVIDER` | `gemini` by default; `local` selects multilingual E5 |
| `EMBEDDING_API_KEY` | Dedicated Google embedding key, or the Google `LLM_API_KEY` |
| `GNANI_API_KEY` | Gnani account key |
| `GNANI_STT_MODEL` | `prisma-v3.0` |
| `GNANI_TTS_MODEL` / `GNANI_TTS_VOICE` | `timbre-v2.5` / `Kaveri` |
| `GNANI_LANGUAGE` | Initial speech locale; `en-IN` |

`LLM_CALLS_ENABLED` does not disable speech or embedding requests. Gemini vectors
are normalized and cached by content hash. Uncached queries use the embedding API;
keys and plaintext queries are not stored in the vector cache. Failures never
silently switch embedding models.

For local CPU embeddings, set `EMBEDDING_PROVIDER=local` in `.env` and build the
matching index with `python -m src index --download-model`. This explicitly selects
quantized `intfloat/multilingual-e5-small`; its model/tokenizer download is about
140 MB. Keep the same provider configured for indexing and querying.

All policy calculations use the dataset reference date, **20 October 2026**,
independently of the host clock.

## Grounding and privacy

Only public KB and catalog content is embedded: 84 sections and 70 product rows.
Search combines semantic similarity, lexical matching, and document authority.
Context includes relevant neighbors, cross-references, and shared escalation rules.
Historical policy exceptions are applied only when authorized by the current policy.

The agent validates the actual reply and cited section IDs, checks digit-form
quantities and calculated dates, and supports explicit abstention. An unambiguous
catalog warranty duration can supplement a reply automatically. Raw model output
is preserved separately; these checks do not prove semantic correctness.

Private order access uses the supplied `lookup_order` tool and caller-provided
proof. Order IDs must contain eight digits. Verification permits two attempts,
is scoped to the session/order, and cannot be restored from chat history. Proofs
are redacted before model context and persistence. Tickets are confirmed only
when the supplied `create_ticket` tool succeeds.

Policy calculations distinguish calendar days, Monday-Saturday business days,
holidays, refund initiation, and settlement. Monetary estimates use Decimal and
half-up rupee rounding. Calculations and eligibility checks do not execute refunds,
cancellations, or pickup bookings; those actions are unavailable through these tools.

## Testing and evaluation

Local checks make no model calls:

```bash
python -m src check --validate
python -m unittest discover -s tests -q
python -m src eval-conversations
```

Regression fixtures expect the supplied dataset at `Dataset/`; the application
also supports an external dataset through configuration or `--dataset-dir`.

Connectivity checks make real provider requests:

```bash
LLM_CALLS_ENABLED=true python -m src smoke --component llm
python -m src smoke --component gnani
python -m src smoke --component gnani --audio /path/to/own-recording.wav
```

Compare prompt-only and grounded answers using the same model, settings, cases,
input variant, and grading procedure:

```bash
LLM_CALLS_ENABLED=true python -m src baseline --stage baseline --input both --judge
LLM_CALLS_ENABLED=true python -m src baseline --stage rag --input both --judge
LLM_CALLS_ENABLED=true python -m src grade runtime/baseline/RUN_ID
python -m src eval-retrieval --input both --k 12
```

Generation receives customer inputs and permitted context, never reference answers
or gold labels. Grading occurs after answers are saved. Runs retain manifests,
source/data hashes, raw output, usage, latency, decisions, used evidence, grades,
and failures. Invalid output is not manually repaired to improve a score. Transport
retries are bounded, and grading refuses to overwrite existing judgments.

The full-comparison script `scripts/final_evaluation.py` supports shared pacing
at 90 weighted quota units per rolling minute, below the configured 100 RPM limit.
It counts retries and embedding items conservatively and honors rate-limit cooldowns.
Token/day quotas can still produce limits. New evaluation runs make API calls;
inspect the shipped evidence when no rerun is needed. The archived capped check
is a single-run experiment with a marker and hard attempt budget.

### Recorded results and limits

The historical full comparison used Gemini 3.5 Flash-Lite, not the current Gemma
configuration. All failures remain in the denominator:

| Input | Prompt-only provisional pass | RAG provisional pass |
| --- | ---: | ---: |
| Original text | 7/52 (13.5%) | 24/52 (46.2%) |
| ASR-like text | 8/52 (15.4%) | 30/52 (57.7%) |

The internal answer-quality target was not met. Model grades are provisional,
without human review, and are not organizer or unseen-test scores.

The final source passed **137 tests** and all **96 conversation decision/tool
checks** across 23 scenarios. Cached retrieval measured **92.2% candidate recall@12**
and **100% enriched-context recall** for both variants. Routing and retrieval
coverage are separate from factual accuracy.

The smaller Gemma 4 26B check retained contract/service failures and an unresolved
grade. The final Gemma 4 31B configuration completed three live contract smoke
cases; a separate zero-call replay through catalog completion passed their required
literal facts. There is no full Gemma accuracy benchmark or human semantic review.

Synthetic English, Hindi, and Hinglish speech checks passed. A mixed batch completed
three valid clips and requested repeats for silence/corruption. Human recordings
and microphone ASR accuracy remain unmeasured. The demo is labelled playback of
actual agent outputs with synthetic caller audio and deterministic policy replies.

## Project layout and artifacts

```text
run.py                     Audio-folder entry point
src/conversation.py        Shared session controller
src/retrieval/             Public index, embeddings, search, and provenance
src/tools/                 Verified lookups, public facts, and ticket gateway
src/rules/                 Policy parsing and deterministic calculations
src/speech/                Gnani adapters, audio handling, and recovery
src/evaluation/            Isolated practice loaders and evaluation runners
src/web.py                 Loopback HTTP server
web/                       Dashboard assets and audio recorder
scripts/                   Evaluation, reporting, and packaging utilities
tests/                     Regression and integration checks
sample_outputs/            Actual saved outputs and evaluation evidence
```

Release artifacts include the [report](report.pdf), [demo](demo.mp4),
[requirement mapping](PS_COMPLETION.md), and `zentromart_voiceops.zip`.
The package excludes Dataset, `.env`, credentials, virtual environments, private
runtime/history, and caches. Generated state belongs under `runtime/`, outside
the dataset. The release remains local; public synchronization and deployment
are separate actions.

To rebuild and verify the package without model calls:

```bash
python scripts/build_release.py
python scripts/verify_submission.py
```

Verification additionally needs Poppler's `pdfinfo` and ffmpeg's `ffprobe`. It checks
required files, hashes, blank example keys, credential exclusions, report page count,
and demo duration/audio. Optional report regeneration needs `reportlab` and DejaVu
Sans: `python scripts/build_report.py`. It reads shipped evidence without API calls.

## Integration references

- [Hosted Gemma models](https://ai.google.dev/gemma/docs/core/gemma_on_gemini_api)
- [Gemini embedding guide](https://ai.google.dev/gemini-api/docs/embeddings)
- [Official Gnani SDK](https://github.com/Gnani-AI-Mintlify/Gnani-Vachana/blob/main/README.md)
- [Gnani Python package](https://pypi.org/project/gnani-vachana/)
- [python-dotenv](https://pypi.org/project/python-dotenv/)
