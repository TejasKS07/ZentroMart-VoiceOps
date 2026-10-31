# Problem statement completion mapping

Updated 1 November 2026. The implementation and local submission artifacts are
complete. Publication is intentionally deferred because the user chose to keep
the release local. The supplied `PS.pdf` is unchanged.

| PS requirement | Implementation and evidence | Status |
| --- | --- | --- |
| Step 1: LLM role/limits and first-turn AI disclosure, p. 8 | `src/agent.py`, baseline prompt and preserved original/ASR outputs in `sample_outputs/evaluation/` | Implemented and measured |
| Step 2: embed documents, retrieve facts, preserve used chunks, p. 9 | Public 154-chunk index; Gemini Embedding 1; authority/history handling; five-field structured response with real section IDs | Implemented |
| Step 1 versus Step 2 comparison, p. 9 | Same-model historical 52-case comparison: 7/52 versus 24/52 original; 8/52 versus 30/52 ASR | Recorded, provisional grades |
| Unknown information must be acknowledged, p. 9 | Unsupported response contract, citation/numeric checks and unknown-delivery smoke case | Implemented; semantic correctness still requires review |
| Step 3: context, clarification, refusal, real tools and handovers, p. 9 | Session controller, verified supplied lookup, ticket idempotency, source-derived dates/amounts and handover summaries | 96/96 decision and tool-name/count checks; 23/23 scenarios |
| Step 4: same agent with Gnani STT/TTS, pp. 10-12 | `src/speech/`, browser record/upload/playback and `run.py`; no alternate speech service | Live synthetic checks passed |
| Voice recovery and readable speech, p. 11 | Spoken identifiers, WAV validation, silence/corruption recovery, audible agent failures and synthesis-only retries | Regression checked |
| Use supplied synthetic data; no real customer calls or secrets, p. 12 | External unchanged Dataset, proof redaction, no private embedding corpus; local `.env` ignored | Checked |
| One command for an audio folder, p. 14 | `LLM_CALLS_ENABLED=true python run.py --input-dir ./test_audio --output-dir ./outputs/run1 --dataset-dir ./Dataset` | Implemented |
| Save text, audio and actual used chunks, p. 14 | Per-clip reply text/WAV, `evidence.json`, sanitized result, timestamps and errors; isolated or manifest sessions | Actual samples shipped |
| Report at most four pages, p. 14 | `report.pdf` includes architecture, both full historical scores, current fixes, failures, bonus dashboard and model limits | Included; verified by submission checker |
| Demo at most two minutes, p. 14 | `demo.mp4`, 27.854 s: grounded refund answer, contextual follow-up, successful handover | Included; synthetic artifact playback explicitly labelled |
| README, run/dependency files, source, blank-key example, samples and ZIP, p. 14 | `zentromart_voiceops.zip`; explicit allowlist, SHA-256 manifest, secret/exclusion and archive checks | Prepared locally |
| Public repository URL, p. 14 | Existing URL: https://github.com/TejasKS07/ZentroMart-VoiceOps; updated branch is local `codex/voiceops-completion` | Public upload deferred by explicit user instruction |

## Current model and bounded live validation

The current language model is `gemma-4-31b-it`, through Google's native text API.
The user authorized at most 20 LLM attempts and 4 Gnani attempts, then allowed
other free models as well. Actual use: **20 LLM attempts, all Gemma 4; 2 Gnani
attempts; zero embedding calls**. The independently measured peak was 10 LLM
requests per rolling minute, below the user's 100 RPM limit.

The first six-case Gemma 4 26B comparison exposed bad citations/claim annotations
and a service failure. It is retained in `sample_outputs/completion/gemma_live/`;
one grade is unresolved, so an overall score is not claimed. The final simpler
contract completed three Gemma 4 31B live smoke cases. A separate frozen-output
replay through the catalog-completion logic passed their required literal-fact
checks without making a new model request. These selected checks do not establish
full practice accuracy or unseen-test performance.

The final source passes **137 automated tests** and all 96 conversation routing
checks. Cached retrieval retains 92.2% candidate recall@12 and 100% enriched-context
recall for both text variants while reducing mean context to 26.4/26.6 chunks.
No question/answer map is embedded in agent code. Public warranty durations and
policy numbers are read from supplied sources; raw model output stays separate
from the assembled agent reply.

## What is deliberately not claimed

- The PS permits testing with one's own voice; it does not require it. No human
  recording was supplied, so microphone ASR accuracy is unmeasured.
- Literal checks and routing checks are not semantic answer accuracy. Historical
  model grades have no human spot-check; the final Gemma model has no full
  52-question benchmark. The earlier 90% internal quality target was not met.
- The demo uses actual agent outputs with synthetic caller speech and deterministic
  policy replies. It does not demonstrate live Gemma policy-answer accuracy.
- No public push, website deployment or external submission was performed. A public
  repository requirement remains deferred by the user's local-only instruction.

## Local verification

Install Python 3.12+, `requirements.txt` and ffmpeg. Keep Dataset external and add
keys only to local `.env`. Start the live dashboard with:

```bash
LLM_CALLS_ENABLED=true python -m src web --live
```

The browser workspace is at http://localhost:8000. To check code and the release
without API calls (the latter also needs Poppler's `pdfinfo`):

```bash
python -m unittest discover -s tests -q
python scripts/verify_submission.py
```

The verifier checks required files, all release-manifest hashes, empty example
keys, credential/private-data exclusions, report pages and demo duration/audio.
No live experiment needs to be rerun to inspect the included evidence.

Primary integration references: [hosted Gemma](https://ai.google.dev/gemma/docs/core/gemma_on_gemini_api),
[official pricing](https://ai.google.dev/gemini-api/docs/pricing#gemma-4),
[Gemini embeddings](https://ai.google.dev/gemini-api/docs/embeddings), and the supplied
Gnani SDK/PS contracts documented in README.
