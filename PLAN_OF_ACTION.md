# VoiceOps 2026 — Plan of action

Prepared: 28 October 2026. Updated: 30 October 2026. Phases 0–6 are implemented locally. Phase 7 local evaluation, four-page report, demo and release package are prepared; public source synchronization is pending. Human-recorded speech testing remains unavailable; the user has no recordings.

## 1. Objective and build priorities

Build ZentroMart's customer-support agent through all four required stages: prompt-only LLM, document-grounded RAG, a stateful conversation with tools, and Gnani speech input/output. The same agent core must serve text, interactive voice, and the submission's audio-folder runner.

Allocate effort according to the rubric:

| Area | Marks | Implementation priority |
| --- | ---: | --- |
| Correct, grounded answers | 25 | Answer from authoritative evidence; abstain when unsupported |
| Retrieval quality | 15 | Preserve section IDs and resolve conflicting documents |
| Conversation handling | 20 | Maintain context, verify callers, clarify, refuse, and escalate |
| Own evaluation | 10 | Preserve reproducible Step 1 versus Step 2 results |
| End-to-end voice | 10 | Process supplied audio and produce spoken replies |
| Voice experience | 10 | Short, clear responses and recovery from unclear audio |
| Report and demo | 10 | Explain results and demonstrate the complete system |

The first milestone is a measurable text agent. Test Gnani connectivity early so account or API issues do not surface at the end. Add a simple recording/playback interface after the batch voice path works. A deployed website or telephone-network integration is not required by the submission interface.

## 2. What is already available

The initial workspace contained the brief and supplied dataset:

- 21 knowledge-base documents, including deliberate policy conflicts and a superseded returns policy.
- 1,000 orders, 1,783 order items, 300 customers, 70 products, 36 pincodes, 4 holidays, and 244 historical support logs.
- 52 single-turn practice questions, including ASR-like text.
- 23 conversations comprising 96 customer turns.
- Working offline `lookup_order` and `create_ticket` helpers.

**Completed check:** `python3 -B Dataset/validate_dataset.py` passes schema, relationships, totals, evidence IDs, and verification gates. This validates the dataset, not an agent's performance.

## 3. Proposed architecture

```text
Customer text ------------------------------+
                                            |
Customer audio -> Gnani STT -> transcript ---+-> Session + agent controller
                                                |-> Evidence retrieval
                                                |-> Verified order / catalog tools
                                                |-> Date and amount calculations
                                                |-> LLM response generation
                                                |-> Ticket creation when required
                                                |
                                                +-> Reply text + evidence trace
                                                +-> Gnani TTS -> reply audio
```

Use Python, one configurable LLM adapter, multilingual embeddings with a local vector index, and a small explicit conversation controller. Start with section-based vector retrieval; add keyword retrieval and reranking only when measured misses justify them. No framework is necessary initially.

Keep policy text and operational facts separate: retrieve policy sections; access private order facts only through the verified tool. Read public catalog and pincode facts through structured lookups, with stable source identifiers. Historical support transcripts must not become policy evidence.

## 4. Execution sequence

The schedule below assumes approximately ten focused working days for one developer. It is an effort estimate, not an organizer deadline; no submission date was found in the brief. API credentials and access are prerequisites for live integration.

| Phase | Estimated effort | Work | Completion gate |
| --- | --- | --- | --- |
  | 0. Foundation | Day 1, first half | Configuration, dataset loader, result schema, evaluation skeleton, LLM and Gnani connectivity checks | Dataset validates; real API smoke checks complete once credentials are available |
| 1. Prompt-only baseline | Day 1, second half | Role/limits prompt, first-turn AI disclosure, run all 52 questions without KB context | Raw outputs and baseline scores saved |
| 2. RAG | Days 2–3 | Parse metadata, preserve sections/tables, embed, retrieve, resolve authority, cite evidence | Re-run identical questions; record improvement and retrieval misses |
| 3. Tools and calculations | Day 4 | Wrap supplied tools, verification control, catalog/pincode lookup, date and refund calculations | Privacy, date boundaries, and failed-action checks pass |
| 4. Conversation | Days 5–6 | Session state, follow-ups, topic/order switching, one-question clarification, escalation | All 23 conversations replay with per-turn results |
| 5. Voice | Day 7 | Gnani STT/TTS adapters, audio normalization, multilingual speech, recovery | Own recorded audio produces grounded text and playable speech |
| 6. Batch runner and demo interface | Day 8 | Audio-folder command, output artifacts, session grouping, recording/playback interface | Fresh process handles a mixed audio batch and saves required evidence |
| 7. Hardening and submission | Days 9–10 | Regression runs, fresh-install check, report, demo, ZIP | Submission checklist complete; reported scores match saved runs |

### Phase 0 — Establish contracts first

- Add environment-based configuration, `.env.example`, dependency pins, and ignore rules for keys, recordings, generated tickets, and runtime outputs.
- Accept an external dataset directory; preserve the original dataset and avoid bundling it in the submission ZIP.
- Keep IDs, phone values, pincodes, SKUs, and refund references as strings; handle blank fields explicitly.
- Set the application reference date to **20 October 2026**, independent of the host clock.
- Define a response object containing decision, reply, evidence IDs, evidence chunks, redacted tool events, and timing. Decisions are `answer`, `clarify`, `refuse`, or `escalate`.
- Run a small live LLM request and a Gnani transcription/synthesis check. Pin the SDK/model/voice combination that actually works.

Implementation status (29 October 2026):

- Offline foundation implemented in `src/`, with setup/commands in `README.md`.
- Supplied dataset validator and application data/practice loaders pass.
- Foundation and Phase 1 regression tests pass; dependency installation was also checked in a clean second virtual environment.
- Live LLM smoke passed with `gemini-3.5-flash-lite`. The originally configured `gemini-3.8-flash` returned HTTP 503 with a provider overload message; both tested Flash-Lite alternatives responded, and the local model was set to `gemini-3.5-flash-lite`.
- Gnani SDK `0.9.0` passed live synthesis and transcription with `timbre-v2.5` / `Kaveri` / `en-IN`. This used synthesized test-speech loopback, not microphone ASR evaluation.
- Both passing checks are recorded in `runtime/smoke/529d2d0e39ae4775a1d9e96191cbf015/checks.json`. **Phase 0 is complete.**
- Real credentials were moved from `.env.example` into ignored `.env` with permissions 0600; the example key fields are blank. A local scan found no configured API keys in source/doc/runtime text artifacts.

### Phase 1 — Preserve a credible baseline

- Introduce the assistant as ZentroMart's AI support assistant on the first reply.
- Establish role, limits, language matching, and concise responses without inserting policy answers into the prompt.
- Run the 52 original questions and save actual, unedited agent outputs.
- Freeze the model, generation settings, and scoring procedure for the Step 2 comparison. Run ASR-like inputs as a separate robustness comparison.
- Keep reference answers, expected decisions, and gold evidence exclusively in evaluation code; never feed them into the answering agent.

Implementation status (29 October 2026):

- Implemented the stateless prompt-only agent, strict response parsing, raw-generation traces and bounded transport retries.
- Added `ask`, `baseline`, and `grade` commands. Baseline runs preserve prompt/model/settings/data/code hashes and save incremental results with partial-run recovery.
- Added a separate semantic-grading pass with explicit provisional scores; it runs after answers are saved and never changes them.
- User subsequently authorized Gemini generation/grading once the entire project is complete. Keep `LLM_CALLS_ENABLED=false` during development; defer original/ASR baseline and RAG generated-answer scoring until the final project run. No further permission is needed for that authorized final scope.

### Phase 2 — Ground answers and resolve policy conflicts

- Parse document headers: type, status, effective date, version, and supersession links.
- Chunk at `[SECTION-ID]` boundaries, carrying parent metadata and intact policy tables. Large sections may use subchunks that retain the original evidence ID.
- Preserve authority: current policy > agent guide > help > FAQ > marketing. Apply the knowledge-base meta rules explicitly.
- Handle superseded material only when a current policy permits it. Example: electronics delivered before 1 August 2026 retain the older window under `RET-3`; do not simply remove all historical content.
- Retrieve across English, Hindi, and Hinglish. Support follow-up query rewriting using relevant session context.
- Combine policy evidence with catalog records when needed, retaining identifiers such as `products.csv:ELE-0012`.
- Distinguish retrieved candidates from evidence actually used. Save both for diagnosis; return the used chunks in submission outputs.
- Ask a clarifying question for missing facts and acknowledge missing information for unsupported questions. Check material factual claims against the supplied evidence before finalizing.
- Re-run the same evaluation and investigate wrong-document retrieval, authority conflicts, missing evidence, and unsupported claims separately.

Phase 2 implementation status (29 October 2026): local quantized multilingual E5 embeddings, 154-chunk persistent vector index, lexical/authority ranking, public catalog entity matching, historical-policy handling, context expansion and a grounded response/claim contract are implemented. Local candidate recall@12 is 88.3% for original text and 91.1% for ASR-like text; enriched-context recall is 98.2% and 100%. These are practice retrieval measurements, not generated-answer scores. See `PHASE_2_3_REPORT.md`.

### Phase 3 — Implement reliable tools and calculations

- Reuse `lookup_order(order_id, verification_value=...)` and `create_ticket(...)` rather than bypassing their behavior.
- On first receipt of an order ID, make the unverified lookup and ask for one proof: registered phone's last four digits or delivery pincode.
- Pass only caller-supplied proof to the tool. Permit one retry, then refuse specific-order disclosure while offering general help.
- Never obtain verification proof by reading customer records on the caller's behalf. Keep proofs out of prompts after verification, logs, saved transcripts, and handover summaries; redact before persisting.
- Bind verification to the current session and order. A change of order requires verification for that order.
- Implement deterministic business-day counting: Monday–Saturday excluding supplied holidays. Return windows use calendar days. Test start-day conventions against policy examples and practice evidence.
- Separate refund initiation from payment settlement, and separate cancellation, return, and return-to-origin refund fields.
- Calculate monetary results from verified records and applicable policy using decimal arithmetic and the dataset's half-up rupee rounding convention. Ask for missing facts rather than infer them, such as an unrecorded grocery slot start time.
- Do not claim cancellation, refund issuance, or pickup booking: the supplied tools provide lookup and ticket creation only. Explain permitted next steps instead.
- Confirm a handover/ticket only after `created=True`; handle tool errors honestly and avoid duplicate tickets on retries.

Phase 3 implementation status (29 October 2026): supplied tools wrapped with per-session/order verification, two-attempt lockout, proof redaction, runtime-only ticket files, persistent ticket idempotency, public lookups and verified operational calculations. Policy values are parsed from the KB, with deterministic dates/Decimal amounts and honest action semantics. All 56 local tests pass across Phases 0–3. Conversation routing remains Phase 4; Gemini calls remain deferred as requested.

### Phase 4 — Make conversation behavior explicit

- Track language, AI introduction, active order, verification attempts, verified facts, intent, missing fields, last answer, relevant evidence, and ticket state.
- Resolve follow-ups such as “how long will that take?” from the active topic. Ask one clear question when ambiguity remains.
- Support language and topic switches without carrying incorrect facts into the new issue.
- Apply all documented escalation triggers: high-value damage, unauthorized activity, legal threats, repeat contacts, overdue refunds, human requests, and safety risks.
- Treat `contacts_last_7_days >= 2` as a third-or-later contact including the current contact, as specified in `ESC-1`.
- Generate handover summaries with the issue, intent, order ID if supplied, relevant facts already provided, checks performed, and escalation reason. Exclude verification secrets.
- Replay each practice conversation in turn order with a fresh session per conversation. The agent sees only customer messages and its own state, not evaluator-only order/customer columns or expected outcomes.
- Test interleaved sessions and order switches for state leakage.

Phase 4 implementation status (29 October 2026): explicit session-scoped conversation
controller, English/Hindi/Hinglish routing, follow-ups, topic/order switches,
one-question clarification, all documented escalation classes, redacted handover
summaries and honest/idempotent ticket handling are implemented. Added `chat` and
`eval-conversations` commands. All 23 practice conversations replay in order:
96/96 completed turns, 96/96 expected decisions, 96/96 expected tool-name/count
matches, and 23/23 scenarios passing both routing checks. All 94 local tests and
the supplied dataset validator pass. These are offline controller measurements;
semantic/generated-answer quality remains unscored and Gemini remains deferred.
Interactive input repair, explicit ID-format validation, `/help`, `/status`, `/reset`
and safe `chat --debug` feedback are also implemented. See `PHASE_4_REPORT.md`. **Phase 4's local implementation/replay gate is complete.**

### Phase 5 — Add Gnani speech around the same core

- Use Gnani exclusively for STT and TTS, with `GNANI_API_KEY` from the environment. Never switch to another speech provider as a fallback.
- Start with file-based transcription and synthesis for the required batch runner. Add streaming only if needed for the final interaction experience.
- Verify SDK calls against a real smoke test; explicitly configure the TTS model, voice, language, and output format. The brief and SDK README differ in some defaults and model listings, so do not depend on defaults.
- Validate audio format and duration; handle empty/silent clips, unclear transcription, timeouts, rate limits, and synthesis failures with bounded recovery.
- Normalize spoken IDs and native-script digits, but ask for confirmation when an important identifier is ambiguous. Do not guess missing digits.
- Keep replies to two or three short sentences; speak amounts in rupees, dates with month and year, and order IDs in groups of four.
- Test English, Hindi, and Hinglish using the developer's own recordings. Use practice `asr_text` for transcript robustness, not as proof that speech recognition works.

Phase 5 implementation status (30 October 2026): Gnani-only STT/TTS, local ffmpeg
normalization, format/size/duration/silence checks, conservative multilingual digit
handling, and bounded recovery are implemented. Live synthetic loopback passed
English, Hindi and Hinglish (3/3). TTS stream headers are repaired from actual PCM
length and validated; synthesis-only retries reuse saved text. The user has no
recordings, so own-recording/microphone recognition is explicitly untested.

### Phase 6 — Build the submission runner as a first-class interface

Implemented command (enable `LLM_CALLS_ENABLED=true` for its default live RAG mode):

```bash
python run.py --input-dir ./test_audio --output-dir ./outputs --dataset-dir ./Dataset
```

For every input, save reply text, reply audio, and machine-readable used evidence with document/section IDs and chunk text. Also save decision, redacted tool results, language, timings, and any error status. Persist transcripts only after redacting verification values.

Process independent files in isolated sessions by default. Define an optional manifest containing `conversation_id`, `turn`, file path, and language for ordered multi-turn audio. This is a proposed local convention: the brief does not specify organizer filenames, session grouping, or an output schema. Document it and adapt if organizers publish a contract.

One failed clip must not stop the rest of the batch. Record failures explicitly; a text-only response after TTS failure is a failed voice turn and must be reported as such. Use unique output names and resumable processing without recreating tickets.

Build a minimal interface for recording/uploading audio, playing the spoken answer, and inspecting retrieved sources for the demo.

Phase 6 implementation status (30 October 2026): `run.py` saves actual text,
Gnani WAV, used chunks, sanitized traces and timings. Optional manifests group
ordered conversations; resume checks fingerprints and avoids duplicate tickets.
A fresh-process mixed batch completed all three valid language clips and marked
silence/corruption as needs-repeat (3 completed, 2 needs-repeat, 0 failed).
A clean installation using an external dataset completed one English clip.
The local no-gradient dashboard includes record/upload, playback, diagnostics,
sources and browser-scoped SQLite chat history with re-verification after restart.
History persistence, upload, desktop and mobile layouts were browser-tested.

### Phase 7 — Freeze evidence and prepare the release

The user selected original plus ASR baseline/RAG with batched grading: 208 answer
calls, 2 query-embedding batches and 24 grading calls, or 234 successful planned
HTTP calls, excluding retries. Gemini Embedding 1 document vectors are already
cached. One shared 90-unit rolling 60-second limiter leaves headroom below the
user's 100 RPM limit, counts embedding items conservatively and pauses both
workers on rate limits. Preserve all failures and provisional grades.

The demo uses actual saved replies and synthetic Gnani caller speech; it is 27.854
seconds, with a grounded refund answer, contextual follow-up and actual handover.
It is labelled artifact playback, not a human microphone recording. The final
report and ZIP must disclose missing human speech tests and unscored holdout and
conversation semantics rather than imply those gates passed.

## 5. Evaluation and release gates

These are proposed internal targets, not official pass thresholds or achieved scores:

| Measure | How to evaluate | Target |
| --- | --- | --- |
| Baseline comparison | Same 52 questions, same model/settings, same scoring | RAG materially improves on prompt-only; explain any regressions |
| Required facts and decisions | Meaning-aware checks against `must_include`, `must_not_include`, and permitted decisions | At least 90% on practice questions |
| Retrieval | Recall@k and evidence precision against `gold_evidence` | At least 90% recall at the chosen k on answerable cases |
| Conversations | Score all 96 turns and complete scenarios; compare required tool calls | At least 90% correct decisions; investigate every failed scenario |
| Privacy/action honesty | Adversarial and failure cases, session/order switches, failed tools | Zero private-order leaks, verification bypasses, or fabricated actions |
| Voice completion | Own recorded English/Hindi/Hinglish batch | Every supported test clip yields text, playable audio, and a trace |
| Recovery | Silent/corrupt audio, API failure, ambiguous ID, failed ticket | No crash, false success, or cross-session disclosure |

Score facts semantically: alternatives separated by ` / ` are equivalent; every fact separated by ` ; ` is required. Accept equivalent dates, spoken numbers, and native-script digits. Do not require English phrasing in Hindi answers. Use deterministic checks for numbers, dates, tool calls, and evidence IDs; manually review or use a consistent judge for paraphrases, with human spot-checks.

Use the supplied practice set for development and create additional paraphrase and edge-case checks from the documents without introducing new policy facts. Keep a small local holdout untouched during tuning. Report practice and holdout results separately; do not imply unseen organizer-test performance.

Record STT, retrieval, LLM, and TTS latency separately, along with token usage and API cost where available. Optimize only after correctness gates pass.

## 6. Suggested source layout

```text
run.py
src/
  config.py
  agent.py
  session.py
  retrieval/       # metadata parsing, chunking, embeddings, search, evidence
  tools/           # provided-helper adapters, public catalog/pincode access
  rules/           # verification, dates, monetary calculations, escalation
  speech/          # Gnani STT/TTS and audio handling
  evaluation/      # isolated practice loader, scoring, experiment reports
tests/              # privacy, date, tool and session regressions
sample_outputs/
README.md
.env.example
requirements.txt
```

Store generated indexes, recordings, tickets, and evaluation runs outside tracked source by default. Dataset files remain external inputs. Adjust paths for the supplied helper's `tools/tickets_created.csv` output and isolate ticket state during evaluation so repeated runs do not contaminate each other.

## 7. Submission checklist

- [x] All four stages implemented; Step 1 and Step 2 measurements retained.
- [ ] Synchronize the updated source and installation/run instructions to the existing public repository (prepared locally).
- [x] Fresh installation can run against an externally supplied dataset.
- [x] Audio-folder runner saves text, speech, and the document chunks used.
- [x] `.env.example` contains variable names and placeholders only.
- [x] Report is at most four pages: architecture and stages, evaluation, failures, improvements, and any bonus work.
- [x] Demo is at most two minutes: grounded spoken answer, contextual follow-up, and a handover or unsupported question.
- [x] Include sample outputs produced by the actual agent without manual editing.
- [x] Package `zentromart_voiceops.zip` with README, source, dependency/run files, report, demo, and sample outputs.
- [x] Exclude API keys, `.env`, virtual environments, and a dataset copy from the ZIP.

## 8. Dependencies, ambiguities, and next action

Required before live development: an LLM choice and credentials, an embedding model choice, and the participant's own Gnani account/key. Configure these behind adapters so a provider change does not affect agent logic.

Resolve source inconsistencies in favor of the updated executable dataset contract: `IDV-2` mentions a `phone_last4` field, but the refined lookup deliberately does not return it. Pass caller proof into `lookup_order`; do not reimplement the stale comparison wording. The dataset README mentions WebSocket STT, while the brief explicitly demonstrates REST; use the brief's file-based submission flow and validate the selected SDK path.

The submission deadline and exact organizer audio/session contract remain unspecified. They do not block foundation, baseline, retrieval, or tool work.

**Release task:** retain final baseline/RAG results, inspect retrieval and contract
failures without rewriting answers, produce the four-page report, and validate the
credential-free release ZIP. Keep the website local for later deployment.

## Sources reviewed

- `PS.pdf`, all 14 pages; especially steps (pages 8–11), rules (12), scoring (13), and submission (14).
- `Dataset/README.md`, `DATA_DICTIONARY.md`, `ZIP_SCHEMA.md`, and `CHANGELOG.md`.
- Dataset validator and supplied Python tool implementation.
- KB authority, identity, conversation, escalation, returns, and refunds documents; inventory and practice schemas/case distributions.
- [Gnani's official Python SDK README](https://github.com/Gnani-AI-Mintlify/Gnani-Vachana/blob/main/README.md), checked 28 October 2026 for the file-based STT/TTS integration path. Actual account/model compatibility still requires a live smoke test.

Final evaluation: 234 successful HTTP requests, 248 attempts including 14 rate-limit responses; peak weighted rolling-minute usage 70. Original provisional baseline/RAG: 7/52 and 24/52; ASR: 8/52 and 30/52. Gemini candidate recall@12: 92.2%; enriched context recall: 100%. Answer-quality target not met. See `PHASE_5_7_REPORT.md` and the frozen sample outputs.
