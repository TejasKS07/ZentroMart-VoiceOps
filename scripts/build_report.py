"""Create the four-page submission report from frozen evaluation evidence.

Authoring dependency: reportlab (not needed to run the support application).
"""
from pathlib import Path
from html import escape
import json
import statistics
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT=Path(__file__).resolve().parents[1]
def read(path):return json.loads(Path(path).read_text())
report=read(read(ROOT/'runtime/final_evaluation/latest.json')['report'])
analysis=read(read(ROOT/'runtime/release_analysis/latest.json')['report'])
assert report['status']=='finished'
font='/usr/share/fonts/truetype/dejavu/DejaVuSans'
pdfmetrics.registerFont(TTFont('Body',font+'.ttf'))
pdfmetrics.registerFont(TTFont('BodyBold',font+'-Bold.ttf'))
pdfmetrics.registerFontFamily('Body',normal='Body',bold='BodyBold')
ink=colors.HexColor('#254f40');muted=colors.HexColor('#637168');light=colors.HexColor('#edf2e9')
styles={
 'title':ParagraphStyle('title',fontName='BodyBold',fontSize=24,leading=30,textColor=ink,spaceAfter=15),
 'sub':ParagraphStyle('sub',fontName='Body',fontSize=10,leading=15,textColor=muted,spaceAfter=15),
 'h':ParagraphStyle('h',fontName='BodyBold',fontSize=12,leading=17,textColor=ink,spaceBefore=12,spaceAfter=6),
 'body':ParagraphStyle('body',fontName='Body',fontSize=9,leading=13.3,spaceAfter=8,textColor=colors.HexColor('#26372d')),
 'small':ParagraphStyle('small',fontName='Body',fontSize=8,leading=11,textColor=muted,spaceAfter=7),
 'cell':ParagraphStyle('cell',fontName='Body',fontSize=8,leading=11),
}
story=[]
def p(text,style='body'):story.append(Paragraph(text,styles[style]))
def heading(text):p(text,'h')
def table(rows,widths):
 t=Table([[Paragraph(escape(str(c)),styles['cell']) for c in row] for row in rows],colWidths=widths,hAlign='LEFT')
 t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),light),('TEXTCOLOR',(0,0),(-1,0),ink),
  ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),
  ('TOPPADDING',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),8),
  ('LINEBELOW',(0,0),(-1,0),1,ink),('LINEBELOW',(0,1),(-1,-1),0.3,colors.HexColor('#d9e1d5'))]))
 story.extend([t,Spacer(1,10)])
def page(title,kicker):
 if story:story.append(PageBreak())
 p(kicker.upper(),'sub');p(title,'title')
def pct(v):return f'{v:.1%}' if v is not None else 'Unresolved'

page('ZentroMart VoiceOps','01 / System and stages')
p('A local multilingual support workspace with grounded policy answers, verified order tools, contextual follow-ups and Gnani speech. Release evidence: 30 October 2026. Policy reference date: 20 October 2026.')
heading('One controller, three interfaces')
table([
 ['Input','Shared support pipeline','Output'],
 ['Text console / dashboard','Session state → intent and verification → public retrieval or verified tools → policy calculations → reply','Text, diagnostics, used source chunks'],
 ['Audio folder / dashboard microphone or upload','Local audio normalization → Gnani STT → same controller → Gnani TTS','Sanitized transcript, WAV reply, evidence and timings'],
],[112,265,122])
heading('Four required stages')
table([
 ['Stage','Implementation'],
 ['1. Prompt-only','Role/limits prompt and customer message only. No policy facts, tools, reference answers or gold evidence.'],
 ['2. RAG','Gemini Embedding 1 (gemini-embedding-001), 768 dimensions; 154 public KB/catalog chunks. Semantic, lexical and authority ranking; original section IDs and current/historical policy rules.'],
 ['3. Conversation + tools','Explicit session state, follow-ups and topic/order switching. Supplied order lookup and ticket helper; deterministic date and Decimal amount calculations.'],
 ['4. Voice','Gnani prisma-v3.0 STT; timbre-v2.5 / Kaveri TTS. English, Hindi and Hinglish; local ffmpeg normalization and conservative spoken-ID handling.'],
],[112,387])
heading('Grounding and identity boundaries')
p('Private orders are never embedded. An order requires caller-supplied proof, with two attempts scoped to session and order. A changed order needs verification again. Proofs are redacted before prompts and persistence. Tickets are confirmed only after successful creation; idempotency prevents duplicate handovers. No cancellation, payment or refund execution tool is invented.')
p('The default dashboard uses deterministic local answers. Live RAG is opt-in; speech always uses Gnani. The site remains loopback-only and has not been deployed.','small')

page('Measured answer quality','02 / Frozen practice evaluation')
p('Same 52 practice questions in original and ASR-like text, same requested Gemini 3.5 Flash-Lite model and generation settings. Provider defaults apply to unspecified temperature/reasoning settings. Each question is an independent session. Reference answers and gold labels stay outside generation; grading occurs only after answers are saved.')
rows=[['Input / stage','Completed','Contract failures','Decision match','Provisional pass']]
for variant in ('original','asr'):
 for stage in ('baseline','rag'):
  r=next(x for x in report['runs'] if x['variant']==variant and x['stage']==stage)
  s,g=r['summary'],r['grading']
  rows.append([variant+' / '+stage,f"{s['completed']}/52",s['failed'],pct(s['decision_match_rate']),f"{g['correct']}/52 ({pct(g['accuracy'])})" if g['accuracy'] is not None else f"≥{g['correct']}/52; {g['unresolved']} unresolved"])
table(rows,[105,70,86,96,142])
p('A provisional pass requires an accepted decision, required facts, no forbidden claims, correct language and first-turn AI disclosure. Contract failures count as failed answers. Grades use batches of up to ten, judged by the same model family; they have not received human review and are not organizer scores.','small')
heading('Retrieval, measured independently')
rows=[['Gemini Embedding 1 / k=12','Original','ASR-like']]
for label,key in [('Mean candidate recall','mean_recall_at_k'),('Mean enriched-context recall','mean_context_recall'),('Mean candidate precision','mean_precision_at_k'),('Mean context chunks / question','mean_context_chunks')]:
 values=[analysis['retrieval'][v][key] for v in ('original','asr')]
 rows.append([label,*[f'{v:.1f}' if key=='mean_context_chunks' else pct(v) for v in values]])
table(rows,[285,107,107])
p('Means use questions with gold evidence where applicable. Context expansion adds neighboring/cross-referenced sections, which increases coverage and prompt size. Retrieval analysis reused cached query vectors and made zero additional provider requests.','small')
heading('Request budget and pacing')
rate=report['rate_limit']
p(f"Approved plan: 208 answer calls + 2 query-embedding batches + 24 judge calls = 234 successful HTTP requests. Actual: {rate['successful_http_requests']} successful, {rate['http_attempts']} attempts, including {rate['rate_limit_responses']} HTTP 429 responses. Shared rolling-window maximum: {rate['peak_quota_weight_in_window']} quota units; safety cap 90 per 60 seconds versus the user's 100 RPM. Embedding items count individually, conservatively. Both workers pause on rate limits and honor retry delays.")

page('Conversation and speech checks','03 / Reliability and local experience')
table([
 ['Check','Observed result','Scope'],
 ['Practice replay','96/96 decisions and tool-name/count checks; 23/23 scenarios','Offline controller routing; semantic answer accuracy unscored'],
 ['Regression suite','124 tests passed','Privacy/session isolation, calculations, tools, recovery, history, HTTP and pacing'],
 ['Synthetic speech loopback','3/3 languages passed','English 2.08 s; Hindi 2.40 s; Hinglish 2.40 s; not human microphone recognition'],
 ['Mixed audio batch','3 completed; 2 needs-repeat; 0 failed','Valid synthetic clips plus corrupt and silent inputs'],
 ['Fresh installation','Pinned dependencies, external dataset and one English audio clip passed','No bundled dataset or runtime dependency cache'],
],[135,174,190])
heading('Recoverable voice turns')
p('Clips are limited to 20 MB and 30 seconds. Invalid/silent audio asks for a repeat without claiming support completion. STT and TTS have bounded retries; a failed synthesis retries the saved text instead of repeating identity checks or creating tickets. Streamed WAV headers are repaired from actual PCM length and validated. The batch saves text, audio, original used chunks, sanitized results, hashes and stage timings; partial batches return a nonzero exit code.')
heading('Dashboard and saved chat history')
p('A simple responsive solid-color dashboard provides text, record/upload, reply playback, sources and per-turn status. Chat history is stored locally in browser-scoped SQLite; it can be reopened or deleted. Verified identity is not persisted. A restored conversation after server restart requires verification again before private order details.')
shot=ROOT/'sample_outputs/dashboard.jpg'
if shot.exists():
 image=Image(str(shot));image.drawWidth=499;image.drawHeight=499*image.imageHeight/image.imageWidth
 if image.drawHeight>155:image.drawHeight=155;image.drawWidth=155*image.imageWidth/image.imageHeight
 image.hAlign='LEFT';story.extend([image,Spacer(1,5)])
p('Browser checks covered saved history, upload/playback controls, desktop and mobile layout. Human recordings were not supplied; microphone ASR remains untested.','small')

page('Failures, limits and reproduction','04 / Release assessment')
heading('Observed failure modes')
rag_runs=[r for r in analysis['runs'] if r['stage']=='rag']
counts={}
for r in rag_runs:
 for reason,n in r['generation_error_counts'].items():counts[reason]=counts.get(reason,0)+n
for reason,n in sorted(counts.items(),key=lambda item:-item[1]):
 p(f'<b>{escape(reason)} ({n} cases).</b> Original generated text is retained; rejected answers were not rewritten or regenerated.','small')
p('Digit-level evidence checks can reject numbers supplied in the question or calculated dates when they are absent from cited chunks. The guard also requires claim text to be an exact reply substring. These checks expose contract failures but do not establish semantic truth. A numeric token appearing in evidence is insufficient proof of a correct policy claim.')
heading('Prioritized improvements')
p('1. Distinguish caller-supplied numbers, deterministic calculations and sourced policy quantities in claim provenance; keep privacy checks strict. 2. Reduce excess context and inspect retrieval misses by language and product ambiguity. 3. Review incomplete answers, language mismatches and AI disclosure failures using the saved per-case grades. 4. Add human English/Hindi/Hinglish recordings and independently reviewed semantic holdout/conversation tests before claiming readiness.')
heading('Limits on the evidence')
p('Practice cases were used during development, so these are not unseen-test estimates. ASR-like text is a robustness comparison, not a speech-recognition benchmark. No human semantic spot-check or extra holdout grading was performed in the approved request budget. Synthetic demo playback uses deterministic policy replies and actual ticket creation; it does not demonstrate live generated-policy quality. Long offline policy extracts may produce a spoken request to narrow the question.')
heading('Run and inspect')
table([
 ['Interface','Command / artifact'],
 ['Dashboard','python -m src web (local) or LLM_CALLS_ENABLED=true python -m src web --live'],
 ['Audio folder','LLM_CALLS_ENABLED=true python run.py --input-dir ./test_audio --output-dir ./outputs/run1 --dataset-dir ./Dataset'],
 ['Evaluation','scripts/final_evaluation.py --asr --judge; explicit billable rerun, not required to inspect saved results'],
 ['Saved evidence','sample_outputs/evaluation/: raw answers, manifests, grades and request audit; sample_outputs/voice_batch/: text, WAV and used chunks'],
],[110,389])
p('Install requirements.txt in Python 3.12+, install ffmpeg and configure keys in local .env. Dataset is external. report.pdf has four pages; demo.mp4 is 27.854 seconds. The release ZIP excludes .env, keys, Dataset, virtual environments and private runtime/history files. Source is prepared locally; public repository synchronization and later website deployment are separate release actions.','small')

def footer(c,doc):
 c.setStrokeColor(colors.HexColor('#d9e1d5'));c.line(48,44,547,44)
 c.setFont('Body',8);c.setFillColor(muted);c.drawString(48,30,'ZentroMart VoiceOps  /  30 October 2026');c.drawRightString(547,30,f'{doc.page} / 4')
doc=SimpleDocTemplate(str(ROOT/'report.pdf'),pagesize=(595,842),leftMargin=48,rightMargin=48,topMargin=42,bottomMargin=57,
 title='ZentroMart VoiceOps — Implementation and evaluation',author='ZentroMart VoiceOps')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print(ROOT/'report.pdf')
