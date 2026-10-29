# Phases 5–7: local implementation and release evidence

Updated: 30 October 2026. Speech, batch, dashboard history and the selected final evaluation are implemented. The site remains locally hosted. Updated source and submission artifacts are prepared locally; the public repository has not been synchronized in this run.

## Final comparison

All four runs finished with 52 questions each and zero unresolved grades. Model: `gemini-3.5-flash-lite`, identical source, model/settings, case selection and practice-file hashes across runs. Unspecified generation settings used provider defaults. No reference answer or gold evidence was supplied to the answering agent. Raw outputs, contract failures and provisional grades are retained without manual edits in `sample_outputs/evaluation/`.

| Input | Stage | Completed | Contract failures | Provisional passes |
| --- | --- | ---: | ---: | ---: |
| original | baseline | 52/52 | 0 | 7/52 (13.5%) |
| original | rag | 41/52 | 11 | 24/52 (46.2%) |
| asr | baseline | 52/52 | 0 | 8/52 (15.4%) |
| asr | rag | 42/52 | 10 | 30/52 (57.7%) |

RAG improved by 32.7 percentage points on original text and 42.3 on ASR-like text. The proposed 90% answer-quality target was not met. Model grades use the same model family, in batches of up to ten, and require independent human review. These are practice scores, not organizer or unseen-test estimates.

## Request accounting

The approved 234 successful HTTP requests comprised 208 answer calls, 2 query-embedding batches and 24 judge calls. Actual transport attempts: **248**, including **14 HTTP 429 responses**. The shared safety cap was 90 quota units per rolling 60 seconds; peak observed and independently recomputed from timestamps was **70**, below the user's 100 RPM. Each embedded text was counted conservatively as a quota unit. Retries were bounded, counted and subject to shared cooldowns. Other provider quotas can still produce rate limits. No extra holdout or conversation-semantic model calls were made.

Document embedding setup occurred before this approved evaluation and is excluded from its 234-call total. All 154 document vectors were already cached. Retrieval and failure diagnosis after evaluation made **zero additional provider calls**, enforced by a network-forbidding embedder in the analysis script.

## Retrieval and failure analysis

Gemini Embedding 1 (`gemini-embedding-001`, normalized 768-dimensional vectors): mean candidate recall@12 **92.2%**, enriched-context recall **100%**, candidate precision **14.7%**, for both original and ASR text on gold-labelled questions. Mean context size was 35.3/35.5 chunks respectively. Wide context improves coverage but adds tokens and distractors; it is not proof of correct answers.

Original RAG rejected 11 outputs: 8 unsupported-number checks and 3 malformed claim structures. ASR RAG rejected 10: 8 unsupported-number checks and 2 numeric replies without claim provenance. The numeric guard can reject dates or caller-provided numbers absent from cited source text. The generation contract also requires exact claim substrings. Neither matching numbers nor a structurally valid citation establishes semantic truth.

Among accepted RAG outputs, the provisional judge found 15 original fact failures and one language failure, and 10 ASR fact failures. Some answers omitted warranty duration or a requested end date; others answered unsupported product/policy questions. Per-case rationales are preserved. Prioritize typed provenance for supplied/calculated/policy values, tighter relevant context and independently reviewed multilingual cases before a new paid evaluation.

## Speech, batch and dashboard

- Gnani-only `prisma-v3.0` STT and `timbre-v2.5` / `Kaveri` TTS; local ffmpeg normalization, 20 MB/30-second input limits, silence/corruption detection and bounded recovery.
- Synthetic English, Hindi and Hinglish STT/TTS loopback: 3/3 passed (2.08, 2.40, 2.40 seconds). These are synthetic checks, not human speech benchmarks.
- Fresh-process mixed batch: 3 completed, 2 needs-repeat, 0 failed. Recovery prompts are not counted as voice completion. Actual text, WAV and used chunks are under `sample_outputs/voice_batch/`.
- `run.py` supports independent files or ordered manifest sessions, source/input fingerprints, resumable output, synthesis-only retries and idempotent tickets.
- Local responsive dashboard: record/upload, reply playback, diagnostics, sources and saved browser-scoped chat history. Server restart requires renewed verification. Upload/history and desktop/mobile layouts were browser-tested; human microphone recognition was not.
- Offline replay: 96/96 expected decisions, 96/96 tool-name/count matches, 23/23 scenarios. Conversation semantic accuracy is unscored.
- 124 automated tests passed. Clean dependency installation, latest-source check against an external dataset and one synthetic English audio command passed.

## Local submission artifacts and limits

`report.pdf` has four visually inspected pages. `demo.mp4` is 27.854 seconds of actual saved agent replies and synthetic caller speech, showing a policy answer, contextual wallet follow-up and successful ticket handover. It is labelled artifact playback and uses deterministic policy replies, not live generated policy answers. `zentromart_voiceops.zip` packages the source and actual sample evidence through an explicit allowlist, excluding Dataset, .env, credentials, private runtime/history and virtual environments.

The user has no recordings, so own-recorded English/Hindi/Hinglish testing is unavailable. Human semantic spot-checks, extra holdout grading, public-repository synchronization and website deployment have not been performed. Deployment remains a later step as requested. Keep these limitations with the measured results.

Release decision: the user explicitly chose to keep the release local. Source is committed on the local `codex/voiceops-completion` branch; nothing was pushed.
