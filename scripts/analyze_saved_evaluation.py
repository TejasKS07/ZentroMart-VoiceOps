"""Analyze frozen outputs and cached vectors without making provider requests."""
from collections import Counter
import json
from pathlib import Path
import statistics
import sys
from datetime import datetime
from uuid import uuid4

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.config import Settings
from src.schemas import GenerationTrace
from src.rag import RagAgent
from src.retrieval.gemini import GeminiEmbedder
from src.retrieval.provider import index_directory
from src.retrieval.index import KnowledgeIndex
from src.evaluation.cases import load_questions
from src.evaluation.retrieval import evaluate_retrieval
from src.evaluation.runner import save_json


def main():
    settings=Settings.load()
    path=Path(json.loads((settings.runtime_dir/'final_evaluation/latest.json').read_text())['report'])
    report=json.loads(path.read_text())
    assert report['status']=='finished','Wait for the authorized evaluation to finish'
    embedder=GeminiEmbedder(settings)
    def no_network(*args,**kwargs):
        raise RuntimeError('Missing cached vector; this analysis forbids network requests')
    embedder._request=no_network
    index=KnowledgeIndex.load(index_directory(settings),embedder,settings.dataset_dir)
    output=settings.runtime_dir/'release_analysis'/uuid4().hex
    output.mkdir(parents=True)
    manifests=[json.loads((Path(r['output_dir'])/'manifest.json').read_text()) for r in report['runs']]
    for key in ('source_sha256','requested_model','generation_parameters','case_ids','practice_sha256'):
        assert all(m[key]==manifests[0][key] for m in manifests), 'Comparison controls differ: '+key
    requests=[json.loads(line) for line in (path.parent/'requests.jsonl').read_text().splitlines()]
    window=[];peak=0
    for request in sorted(requests,key=lambda r:r['started_at']):
        t=datetime.fromisoformat(request['started_at']).timestamp()
        window=[(started,weight) for started,weight in window if started>t-60]
        window.append((t,request['quota_weight']));peak=max(peak,sum(w for _,w in window))
    assert peak<=90, 'Request audit exceeded the safety cap'
    analysis={'retrieval':{},'runs':[], 'additional_provider_requests':0,
        'comparison_controls':'same source, practice file, model, generation settings and case IDs',
        'independently_verified_peak_quota_units':peak}
    for variant in ('original','asr'):
        summary=evaluate_retrieval(index,settings.dataset_dir,output/('retrieval_'+variant),k=12,asr=variant=='asr')
        assert summary['embedding_requests']==0
        analysis['retrieval'][variant]=summary
    for run in report['runs']:
        directory=Path(run['output_dir'])
        rows=[json.loads(line) for line in (directory/'results.jsonl').read_text().splitlines()]
        judgements=[json.loads(line) for line in (directory/'judgements.jsonl').read_text().splitlines()]
        latencies=[r.get('response',{}).get('generation',r.get('generation',{})).get('latency_ms') for r in rows]
        latencies=[v for v in latencies if isinstance(v,(int,float))]
        reasons=[]
        if run['stage']=='rag':
            cases={c.case_id:c for c in load_questions(settings.dataset_dir,asr=run['variant']=='asr')}
            for row in rows:
                if row['status']!='failed' or 'generation' not in row:continue
                trace=GenerationTrace(**row['generation'])
                class SavedClient:
                    def complete(self,messages,**kwargs):return trace
                captured=[]
                def observe(frame,event,arg):
                    if event=='exception' and frame.f_code.co_filename.endswith('/src/rag.py'):
                        kind,value,_=arg
                        if kind is ValueError:captured.append(str(value))
                    return observe
                sys.settrace(observe)
                try:RagAgent(SavedClient(),index).respond(cases[row['case_id']].customer)
                except ValueError:pass
                finally:sys.settrace(None)
                reasons.append({'case_id':row['case_id'],'reason':captured[0] if captured else 'response_contract'})
        checks=Counter(k for row in judgements for k,v in row.get('checks',{}).items() if v is False)
        analysis['runs'].append({'stage':run['stage'],'variant':run['variant'],
            'generation_error_counts':dict(Counter(r['reason'] for r in reasons)),
            'generation_errors':reasons,'failed_judge_checks':dict(checks),
            'median_generation_ms':statistics.median(latencies) if latencies else None})
    assert embedder.network_requests==0
    save_json(output/'summary.json',analysis)
    save_json(settings.runtime_dir/'release_analysis/latest.json',{'report':str(output/'summary.json')})
    print(json.dumps({'output_dir':str(output),**analysis},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
