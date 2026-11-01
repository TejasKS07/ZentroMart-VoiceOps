# Actual saved outputs

Voice clips and demo caller audio were synthesized by Gnani. No human recordings were supplied. Replies were produced by the actual agent, without manual answer edits. The voice batch uses deterministic local policy replies; it is separate from the live Gemini generation comparison.

`voice_batch/` contains 3 completed language clips and 2 needs-repeat recovery cases. Each completed clip has the reply WAV, text and original used evidence. `conversations/` records offline routing/tool checks, not semantic accuracy. `evaluation/` preserves final baseline/RAG original/ASR model outputs and provisional grades; failures remain included. `analysis/` measures cached retrieval and diagnoses frozen outputs with zero extra model calls. `dashboard.jpg` shows the local UI and saved chat history. No keys, dataset copy or private history database is included.
