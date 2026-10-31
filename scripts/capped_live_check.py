"""Explicitly authorized Gemma 4 check: at most 20 LLM and 4 Gnani attempts.

Uses a six-question targeted comparison, not a new full benchmark. Embedding
network access is forbidden. The one-request canary shares the same total cap.
"""
from dataclasses import replace
from pathlib import Path
import argparse
import hashlib
import json
import sys
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.config import Settings
from src.agent import BaselineAgent,BASELINE_PROMPT,BASELINE_PROMPT_VERSION
from src.rag import RagAgent,RAG_PROMPT,RAG_PROMPT_VERSION
from src.llm import ChatCompletionClient
from src.rate_limit import WeightedRateLimiter
from src.dataset import PublicKnowledge
from src.retrieval.gemini import GeminiEmbedder
from src.retrieval.index import KnowledgeIndex
from src.retrieval.provider import index_directory
from src.evaluation.cases import load_questions
from src.evaluation.runner import run_questions,save_json
from src.evaluation.judge import grade_run_batched
from src.conversation import ConversationAgent
from src.tools.operations import SupportOperations
from src.speech.provider import GnaniSpeech
from src.speech.engine import VoiceEngine
from src.speech.audio import SpeechError

IDS=('PQ0002','PQ0004','PQ0020','PQ0022','PQ0054','PQ0057')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true',help='Makes the authorized billable requests')
    args=parser.parse_args()
    if not args.execute:parser.error('Explicit --execute is required')
    settings=replace(Settings.load(),llm_calls_enabled=True,llm_model='gemma-4-26b-a4b-it',llm_reasoning_effort='minimal')
    output=Path(json.loads((settings.runtime_dir/'gemma_completion/latest.json').read_text())['output_dir'])
    canary=json.loads((output/'canary.json').read_text())
    assert canary['status']=='completed' and canary['model']==settings.llm_model
    # Exclusive marker prevents restarting an interrupted run with a fresh budget.
    with (output/'execution_started.txt').open('x') as stream:stream.write(datetime.now(timezone.utc).isoformat())
    prior=[json.loads(line) for line in (output/'requests.jsonl').read_text().splitlines()]
    assert len(prior)<20
    limiter=WeightedRateLimiter(max_attempts=20-len(prior),log_path=output/'requests.jsonl')
    client=ChatCompletionClient(settings,request_limiter=limiter);client.max_attempts=2
    embedder=GeminiEmbedder(settings)
    def forbid(*args,**kwargs):raise RuntimeError('Embedding network calls are forbidden in this check')
    embedder._request=forbid
    index=KnowledgeIndex.load(index_directory(settings),embedder,settings.dataset_dir)
    knowledge=PublicKnowledge.load(settings.dataset_dir)
    rag=RagAgent(client,index,knowledge=knowledge)
    by_id={c.case_id:c for c in load_questions(settings.dataset_dir)}
    cases=tuple(by_id[i] for i in IDS)
    source={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'src').rglob('*')) if p.suffix in {'.py','.txt'}}
    secrets=(settings.llm_api_key,settings.embedding_api_key,settings.gnani_api_key)
    report={'model':settings.llm_model,'generation_parameters':settings.generation_parameters(),
        'scope':'targeted six-question original-text comparison; not a full practice estimate',
        'approved_gemma_attempts':20,'approved_gnani_attempts':4,'runs':[],'embedding_http_attempts':0}
    speech=None
    try:
        for stage,agent,prompt,version in [('baseline',BaselineAgent(client),BASELINE_PROMPT,BASELINE_PROMPT_VERSION),('rag',rag,RAG_PROMPT,RAG_PROMPT_VERSION)]:
            directory=output/stage
            summary=run_questions(cases,agent.respond,directory,secrets=secrets,
                metadata={'stage':stage,'case_selection':'targeted regression cases; frozen before generation',
                    'requested_model':settings.llm_model,'generation_parameters':settings.generation_parameters(),
                    'source_sha256':source,'prompt_version':version,'system_prompt':prompt,'input_variant':'original',
                    'embedding_mode':'cached vectors only; zero embedding API calls'},
                progress=lambda text:print(stage+': '+text,flush=True))
            row={'stage':stage,'summary':summary}
            if summary['run_status']=='finished':row['grading']=grade_run_batched(directory,cases,client,batch_size=10,secrets=secrets)
            report['runs'].append(row)
            save_json(output/'comparison_summary.json',report)
        class BudgetedSpeech(GnaniSpeech):
            attempts=0
            def _call(self,stage,callback):
                def bounded():
                    if self.attempts>=4:raise SpeechError('budget_exhausted',stage,'Approved speech-attempt budget exhausted')
                    self.attempts+=1
                    with (output/'speech_requests.jsonl').open('a') as stream:
                        stream.write(json.dumps({'stage':stage,'attempt':self.attempts,'started_at':datetime.now(timezone.utc).isoformat()})+'\n')
                    return callback()
                return super()._call(stage,bounded)
        speech=BudgetedSpeech(settings)
        agent=ConversationAgent(SupportOperations(settings.dataset_dir,settings.runtime_dir/'completion_tools'),rag=rag)
        engine=VoiceEngine(agent,speech)
        # A real spoken handover exercises STT, session routing, the ticket tool
        # and TTS without an unnecessary model call or a new embedding request.
        turn=engine.process(settings.runtime_dir/'voice_inputs/demo/03.wav','gemma-completion-'+output.name,'en-IN')
        directory=output/'voice';directory.mkdir()
        save_json(directory/'result.json',turn.to_dict())
        if turn.audio:(directory/'reply.wav').write_bytes(turn.audio)
        if turn.response:
            (directory/'reply.txt').write_text(turn.response.reply+'\n')
            save_json(directory/'evidence.json',{'evidence':turn.response.to_dict()['evidence']})
        report['voice']={'status':turn.status,'voice_complete':turn.to_dict()['voice_complete'],'language':'en-IN',
            'kind':'synthetic spoken handover through the live-configured controller; this tool turn needs no LLM'}
        report['status']='finished'
    except Exception as exc:
        report.update(status='incomplete',error_type=type(exc).__name__)
    finally:
        requests=[json.loads(line) for line in (output/'requests.jsonl').read_text().splitlines()]
        assert len(requests)<=20 and embedder.network_requests==0
        report['gemma_http_attempts']=len(requests)
        report['successful_gemma_http_requests']=sum(r['status']==200 for r in requests)
        report['gnani_api_attempts']=speech.attempts if speech else 0
        report['request_limiter']=limiter.summary()
        save_json(output/'comparison_summary.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
