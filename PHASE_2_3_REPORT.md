# Phases 2 and 3 implementation report

Updated 29 October 2026. Implementation and local verification are complete for
these phases. Generated-answer validation is deliberately deferred until the full
project is complete, following the user's instruction. Gemini calls in this work: **0**.

## Phase 2 — Grounded retrieval

- Local CPU embeddings: quantized multilingual E5-small, 384 dimensions, via
  FastEmbed 0.9.1 / ONNX Runtime. The public weights/tokenizer download is cached
  under `runtime/models`; no dataset content is sent for embedding.
- Persistent index: **154 chunks**, comprising **84 original KB sections** and
  **70 catalog rows**. Tables, section IDs and document metadata are preserved.
- Private order/customer/support-log data and evaluation answers are excluded.
- Semantic and lexical ranking, authority preference, public product-name/SKU
  matching, explicitly uncertain ASR SKU candidates, historical delivery dates,
  current-successor rules, cross-references and policy neighbors are implemented.
- Index reload checks KB/catalog fingerprints, model identity, weights and tokenizer.
- The RAG generator contract preserves actual used evidence and claim provenance,
  distinguishes candidates/context from used sources, rejects fabricated source
  IDs and unsupported digit-form numbers, and permits abstention for missing information.

Local practice retrieval measurements, with **k=12**:

| Metric | Original text | ASR-like text |
| --- | ---: | ---: |
| Questions processed | 52 | 52 |
| Questions with gold evidence | 47 | 47 |
| Mean candidate recall@12 | 88.3% | 91.1% |
| Mean enriched-context recall | 98.2% | 100.0% |
| Mean candidate precision against gold IDs | 14.0% | 14.5% |
| Mean context precision against gold IDs | 5.3% | 5.3% |
| Mean context size | 36.1 chunks | 36.6 chunks |

Context includes policy neighbors, explicit links, KB rules and agent guides, so
its recall must not be confused with top-12 recall. Gold evidence lists expected
answer sources, not every potentially useful guardrail. The larger context is a
token-cost tradeoff to measure in the final live run. These practice measurements
were used during development and are not holdout/generalization claims. No answer
accuracy, baseline-vs-RAG improvement, or actual speech-recognition accuracy is
reported here.

Latest source artifacts:

- `runtime/retrieval_evaluations/original-c82f09d64678/summary.json`
- `runtime/retrieval_evaluations/asr-8cf845e81a34/summary.json`
- Each folder also contains per-question retrieved, context, expected and missing IDs.

## Phase 3 — Tools and rules

- Executes the original provided helper implementations in an isolated module;
  runtime ticket output is redirected without editing the dataset.
- Withholds order facts until a caller-supplied phone suffix or delivery pincode
  succeeds. Allows one retry, tracks attempts across order switches, isolates
  sessions, and rejects a backend result for a different order.
- Never supplies expected verification values to the model. Sanitizes known proof
  values, email and full-phone text in summaries/exports. Does not treat backend
  failure as an incorrect proof.
- Ticket handover returns success only after a real `created=True` result.
  Persistent idempotency avoids duplicate successful tickets on repeated requests.
  Failures remain retryable and never become fabricated confirmations.
- Public catalog/pincode facts preserve original source rows. Unknown pincodes
  remain unknown, rather than being declared unserviceable.
- Reads return windows, historical/festive exceptions, pickup fees, refund method
  ranges and escalation thresholds from KB sections. Business-day calculations
  include Saturdays and skip Sundays/CSV holidays; return windows use calendar days.
- Separates return, cancellation and RTO refund fields and preserves RRN values.
  Distinguishes initiation from settlement and does not invent a precise wallet
  deadline from a date-only record.
- Uses Decimal half-up rupee rounding, line/quantity selection, proportional coupon
  allocation, minimum-order clawback, pickup waivers and whole-order fee rules.
- Missing grocery slot/time produces clarification. Cancellation methods guide
  customer actions; no cancellation/refund/pickup execution tool is invented.
- Escalation threshold calculations cover high-value damage, repeat contacts,
  overdue refunds and explicit human/security/legal/safety flags. Natural-language
  intent detection and conversation orchestration are Phase 4 responsibilities.

## Verification and remaining integration

**56 local tests pass**, covering previous foundations plus policy precedence,
index persistence/staleness, historical applicability, ASR ambiguity, evidence and
numeric-claim failures, abstention, real supplied-tool verification, failed proofs,
native-script pincodes, session isolation, ticket creation/redaction/idempotency,
backend failure recovery, holiday/date boundaries, coupon/quantity calculations,
fee exceptions and policy updates changing computed results.

The checks use real local embeddings for retrieval evaluation and test doubles
for generation. Structural/numeric claim checks do not prove semantic entailment;
actual Gemini answer quality, abstention quality, latency and cost remain to be
measured after the complete project is built. Do not treat these local tests as
52 scored agent answers or as a finished conversation/voice implementation.

## Commands

```bash
.venv/bin/python -m unittest discover -s tests -q
.venv/bin/python -m src check --validate
.venv/bin/python -m src index
.venv/bin/python -m src retrieve "UPI refund ka time kitna hai?"
.venv/bin/python -m src eval-retrieval --input both --k 12
```

The live-generation switch is `LLM_CALLS_ENABLED=false` in local `.env` and the
example template. Keys remain in ignored `.env`; no Gemini request was made.

Embedding implementation references: [FastEmbed](https://github.com/qdrant/fastembed/blob/main/README.md),
[multilingual E5 model card](https://huggingface.co/intfloat/multilingual-e5-small).
