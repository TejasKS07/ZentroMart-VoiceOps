"""Build the four-page PS report entirely from shipped, frozen sample evidence.

Authoring needs reportlab and DejaVu Sans; the support app does not need reportlab.
"""
from pathlib import Path
from html import escape
import json
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT=Path(__file__).resolve().parents[1]
def read(path):return json.loads(Path(path).read_text())
history=read(ROOT/'sample_outputs/evaluation/summary.json')
completion=ROOT/'sample_outputs/completion'
account=read(completion/'accounting.json')
comparison=read(completion/'gemma_live/comparison_summary.json')
smoke=read(completion/'gemma_live/final_smoke.json')
retrieval={v:read(completion/'retrieval'/v/'summary.json') for v in ('original','asr')}
assert history['status']=='finished'
font='/usr/share/fonts/truetype/dejavu/DejaVuSans'
pdfmetrics.registerFont(TTFont('Body',font+'.ttf'));pdfmetrics.registerFont(TTFont('BodyBold',font+'-Bold.ttf'))
pdfmetrics.registerFontFamily('Body',normal='Body',bold='BodyBold')
ink=colors.HexColor('#254f40');muted=colors.HexColor('#637168');light=colors.HexColor('#edf2e9')
styles={
 'title':ParagraphStyle('title',fontName='BodyBold',fontSize=23,leading=29,textColor=ink,spaceAfter=13),
 'sub':ParagraphStyle('sub',fontName='Body',fontSize=9,leading=14,textColor=muted,spaceAfter=12),
 'h':ParagraphStyle('h',fontName='BodyBold',fontSize=12,leading=17,textColor=ink,spaceBefore=11,spaceAfter=6),
 'body':ParagraphStyle('body',fontName='Body',fontSize=9,leading=13,spaceAfter=8,textColor=colors.HexColor('#26372d')),
 'small':ParagraphStyle('small',fontName='Body',fontSize=8,leading=11,textColor=muted,spaceAfter=7),
 'cell':ParagraphStyle('cell',fontName='Body',fontSize=8,leading=11)}
story=[]
def p(text,style='body'):story.append(Paragraph(text,styles[style]))
def h(text):p(text,'h')
def page(title,kicker):
 if story:story.append(PageBreak())
 p(kicker.upper(),'sub');p(title,'title')
def table(rows,widths):
 t=Table([[Paragraph(escape(str(c)),styles['cell']) for c in row] for row in rows],colWidths=widths,hAlign='LEFT')
 t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),light),('VALIGN',(0,0),(-1,-1),'TOP'),
  ('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),
  ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),
  ('LINEBELOW',(0,0),(-1,0),1,ink),('LINEBELOW',(0,1),(-1,-1),0.3,colors.HexColor('#d9e1d5'))]))
 story.extend([t,Spacer(1,8)])
def pct(v):return f'{v:.1%}' if v is not None else 'Unresolved'

page('ZentroMart VoiceOps','01 / Completed local implementation')
p('Release: 1 November 2026. Policy reference date: 20 October 2026. One support core serves the text console, local dashboard and audio-folder submission command. The current hosted language model is Gemma 4 31B; speech uses Gnani exclusively. The release remains local at the user\'s request.')
h('Four required stages')
table([
 ['Stage','Implementation'],
 ['1. Prompt-only LLM','Role/limits prompt, AI disclosure and customer message. No company documents, tools or answer keys. Actual outputs are retained for comparison.'],
 ['2. Document-grounded RAG','154 public KB/catalog chunks; Gemini Embedding 1, normalized 768-dimensional vectors; semantic, lexical and authority ranking. Current policies, historical exceptions and original section IDs are preserved.'],
 ['3. Conversation and tools','Session/order-scoped verification, one-question clarification, contextual follow-ups, topic switches, deterministic calendar/amount calculations, refusals and idempotent human handovers.'],
 ['4. Gnani voice','prisma-v3.0 STT and timbre-v2.5 / Kaveri TTS. Audio normalization, English/Hindi/Hinglish, conservative identifier handling, playback and audible recovery.'],
],[111,388])
h('Shared pipeline')
table([
 ['In','Brain','Out'],
 ['Text or Gnani transcript','Session controller -> public retrieval / verified lookup -> calculations -> reply','Text, decision, diagnostics and original used chunks'],
 ['Audio folder or browser audio','Local ffmpeg -> Gnani STT -> same controller -> Gnani TTS','Reply WAV, sanitized transcript, evidence, stage timings and explicit failures'],
],[104,262,133])
h('Identity, policy and action boundaries')
p('Private orders, customers and historical call logs are not embedded. Order access requires caller-supplied proof, with two attempts per session/order. A changed order requires verification again. Proofs are redacted before model context and persistence; saved history never restores verified identity. Tickets are confirmed only after the supplied tool succeeds. The agent cannot execute cancellation, refund issuance or pickup booking.')
p('No practice question/answer map is used. Policy numbers come from KB sections; public catalog durations and deterministic dates may supplement model output automatically. Raw model text is preserved separately from the actual assembled agent reply.','small')

page('Evaluation and what changed','02 / Historical benchmark and bounded follow-up')
p('The preserved 30 October comparison used the same Gemini 3.5 Flash-Lite model, generation settings and 52 cases per variant. Reference answers and labels were excluded from generation and used only afterward for grading. These historical scores are not scores for the current Gemma configuration.')
rows=[['Input / stage','Completed','Rejected outputs','Provisional pass']]
for variant in ('original','asr'):
 for stage in ('baseline','rag'):
  r=next(x for x in history['runs'] if x['variant']==variant and x['stage']==stage)
  rows.append([variant+' / '+stage,f"{r['summary']['completed']}/52",r['summary']['failed'],f"{r['grading']['correct']}/52 ({pct(r['grading']['accuracy'])})"])
table(rows,[140,92,113,154])
p('RAG improved by 32.7 percentage points on original text and 42.3 on ASR-like text, while missing the proposed 90% target. Contract failures remain in the denominator. The batched model judge uses required facts, forbidden claims, language, disclosure and accepted decisions; it is provisional, without human review.','small')
h('Gemma-only follow-up within the approved cap')
p('A six-question targeted check used Gemma 4 26B for both stages. Baseline: 6 completed, 1 provisional pass. RAG: 1 completed, 4 contract rejections and 1 service failure; 1 provisional pass and 1 unresolved grade, so no overall accuracy is reported. Rejections included invented source paths, duplicate or mismatched claim annotations; these runs remain unchanged.')
p('The final contract was simplified to validate the actual reply, section IDs, supported/unsupported status, numbers and dates. Gemma 4 31B then completed all 3 live smoke cases (warranty, unsupported delivery and refund date). A subsequent zero-call replay through catalog completion passed their required literal-fact checks. These selected checks are not a new practice-accuracy estimate.','small')
h('Current retrieval with less unrelated context')
rows=[['Gemini Embedding 1 / k=12','Original','ASR-like']]
for label,key in [('Candidate recall','mean_recall_at_k'),('Enriched-context recall','mean_context_recall'),('Mean context chunks','mean_context_chunks')]:
 rows.append([label,*[f'{retrieval[v][key]:.1f}' if key=='mean_context_chunks' else pct(retrieval[v][key]) for v in ('original','asr')]])
table(rows,[285,107,107])
p('Context fell from 35.3/35.5 to 26.4/26.6 chunks while retaining labelled evidence coverage. Shared safety/escalation rules stay available; unrelated agent guides are no longer added unconditionally. Retrieval rechecks used cached vectors and made no provider calls.','small')

page('Voice, conversation and dashboard','03 / Verified behavior and local experience')
table([
 ['Check','Observed result','Limit'],
 ['Final regression suite','137 tests passed','Contracts, privacy, dates, tools, history, HTTP, budgets and audible failure recovery'],
 ['Practice conversations','96/96 decisions and tool-name/count matches; 23/23 scenarios','Routing checks, not semantic answer accuracy'],
 ['Synthetic language loopback','English, Hindi and Hinglish: 3/3 passed','Synthesized caller audio; not human recognition testing'],
 ['Mixed audio folder','3 completed; 2 needs-repeat; 0 failed','Silence/corruption request repeats and do not count as support completion'],
 ['New spoken handover','STT -> controller -> actual ticket -> TTS completed','2 Gnani attempts; this tool turn needs no LLM'],
 ['Clean execution','Pinned dependencies and external-dataset checks; isolated ZIP validation','Dataset, secrets and runtime history are external'],
],[125,181,193])
h('Useful and honest recovery')
p('The runner validates 20 MB / 30-second clips, normalizes mono 16 kHz PCM and saves text, WAV, used chunks, traces and timings. Silent or ambiguous input asks for a repeat. A model/tool failure now also produces an audible explanation while remaining a failed support turn. Synthesis-only retries reuse text and preserve failure status; they do not repeat verification or ticket actions. Gnani streaming WAV headers are validated and repaired from actual PCM length.')
h('Local dashboard and saved conversations')
p('The responsive solid-color workspace offers text, record/upload, playback, source inspection and per-turn diagnostics. Browser-scoped SQLite history can be reopened or deleted; a restored chat after server restart requires verification again. The live status identifies gemma-4-31b-it. No deployment or public push was performed.')
shot=completion/'dashboard_final.jpg'
if shot.exists():
 im=Image(str(shot));im.drawHeight=135;im.drawWidth=135*im.imageWidth/im.imageHeight;im.hAlign='LEFT';story.extend([im,Spacer(1,5)])
p('The 27.854-second demo uses actual saved agent replies and synthetic Gnani caller audio: refund policy, contextual wallet follow-up and successful handover. It is labelled artifact playback, uses deterministic policy replies and does not claim to demonstrate Gemma answer accuracy.','small')

page('Reproduction and release assessment','04 / PS deliverables, budget and limits')
h('Implemented hardening')
p('Dates and equivalent numeric forms are normalized before checks; refund settlement dates use the KB calendar rather than model arithmetic. Recognized caller dates may be echoed without licensing invented amounts or durations. Public warranty duration is read from one unambiguous catalog record; contradictory explicit durations are rejected. Reply-level evidence replaces brittle duplicated claim text. Unknown citations and unsupported numeric claims still fail safely; raw outputs and failed grading batches remain available.')
h('Approved API accounting')
p(f"Follow-up: {account['llm_attempts']}/20 authorized LLM attempts, {account['successful_llm_http_requests']} HTTP successes, {account['rate_limit_responses']} rate-limit responses and {account['connection_failures']} connection failure. All used Gemma 4. Peak: {account['peak_llm_requests_per_60_seconds']} requests per rolling minute, below 100 RPM; the final 31B calls were paced at one per minute. Gnani: {account['gnani_attempts']}/4 attempts. Embedding API calls: zero. No further model requests were made after the cap.")
p('The earlier 234-success benchmark used 248 attempts, including 14 HTTP 429 responses, with a weighted peak of 70 per minute. Its request budget and results are separate from the new capped check. Gemma input/output is currently listed as free by Google; speech and future uncached embeddings have separate account terms.','small')
h('Exact run commands')
table([
 ['Use','Command'],
 ['Live local dashboard','LLM_CALLS_ENABLED=true python -m src web --live'],
 ['Required audio-folder interface','LLM_CALLS_ENABLED=true python run.py --input-dir ./test_audio --output-dir ./outputs/run1 --dataset-dir ./Dataset'],
 ['No-API verification','python -m unittest discover -s tests -q; python scripts/verify_submission.py'],
],[126,373])
p('Use Python 3.12+, requirements.txt and ffmpeg. Configure local .env keys; keep the supplied Dataset external. Report generation additionally uses reportlab and verification uses pdfinfo/ffprobe. sample_outputs/ includes frozen comparisons, current smoke/replay evidence, actual voice artifacts and diagnostics. The ZIP uses an explicit allowlist, hashes and a credential scan.','small')
h('Scope and release limits')
p('Digit/source checks do not prove semantic correctness. The final Gemma model has not received a full 52-question benchmark or human semantic review; selected smoke and literal checks cannot replace them. No organizer/unseen-test score is claimed. The brief permits synthetic data and optional own-voice tests; the user supplied no recordings. The original problem statement is unchanged. A public repository is requested by the PS, but the user explicitly chose a local release, so publishing is deferred. The local code, four-page report, short demo and ZIP are prepared.')
p('References: Google hosted Gemma guide (ai.google.dev/gemma/docs/core/gemma_on_gemini_api), official API pricing (ai.google.dev/gemini-api/docs/pricing), supplied PS and Dataset contracts. Full source-to-requirement mapping: PS_COMPLETION.md.','small')

def footer(c,doc):
 c.setStrokeColor(colors.HexColor('#d9e1d5'));c.line(48,44,547,44);c.setFont('Body',8);c.setFillColor(muted)
 c.drawString(48,30,'ZentroMart VoiceOps  /  1 November 2026');c.drawRightString(547,30,f'{doc.page} / 4')
SimpleDocTemplate(str(ROOT/'report.pdf'),pagesize=(595,842),leftMargin=48,rightMargin=48,topMargin=42,bottomMargin=57,
 title='ZentroMart VoiceOps - PS implementation and evaluation',author='ZentroMart VoiceOps').build(story,onFirstPage=footer,onLaterPages=footer)
print(ROOT/'report.pdf')
