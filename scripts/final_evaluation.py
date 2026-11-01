"""Final Gemini comparison. Scope is explicit; extras are never enabled by default."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Lock
from uuid import uuid4
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.config import Settings
from src.evaluation.experiment import baseline
from src.evaluation.runner import save_json
from src.retrieval.provider import create_embedder,index_directory
from src.retrieval.index import KnowledgeIndex
from src.rate_limit import WeightedRateLimiter
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--limit',type=int,help='Optional small smoke comparison')
parser.add_argument('--judge',action='store_true',help='Additional Gemini grading calls; requires approval')
parser.add_argument('--asr',action='store_true',help='Additional ASR comparison; requires approval')
args=parser.parse_args()
settings=replace(Settings.load(),llm_calls_enabled=True)
output=settings.runtime_dir/'final_evaluation'/uuid4().hex;output.mkdir(parents=True)
limiter=WeightedRateLimiter(capacity=90,log_path=output/'requests.jsonl')
save_json(settings.runtime_dir/'final_evaluation'/'latest.json',{'report':str(output/'summary.json')})
report={'status':'running','model':settings.llm_model,'generation_parameters':settings.generation_parameters(),
 'scope':{'original':True,'asr':args.asr,'automatic_grading':args.judge,'judge_batch_size':10,'limit':args.limit},'runs':[]};lock=Lock()
def record(row):
 with lock:report['runs'].append(row);report['rate_limit']=limiter.summary();save_json(output/'summary.json',report);print(json.dumps(row),flush=True)
def run_stage(stage):
 index=KnowledgeIndex.load(index_directory(settings),create_embedder(settings,request_limiter=limiter),settings.dataset_dir) if stage=='rag' else None
 for variant in (('original','asr') if args.asr else ('original',)):
  path,summary=baseline(settings,dataset_dir=settings.dataset_dir,variant=variant,limit=args.limit,judge=args.judge,stage=stage,index=index,request_limiter=limiter,progress=lambda text:print(f'{stage}/{variant}: {text}',flush=True))
  grading=json.loads((path/'grading_summary.json').read_text()) if (path/'grading_summary.json').exists() else None
  record({'stage':stage,'variant':variant,'output_dir':str(path),'summary':summary,'grading':grading})
  if summary['run_status']!='finished':return
try:
 with ThreadPoolExecutor(max_workers=2) as pool:
  futures=[pool.submit(run_stage,stage) for stage in ('baseline','rag')]
  for future in futures:future.result()
 report['status']='finished' if len(report['runs'])==(4 if args.asr else 2) and all(r['summary']['run_status']=='finished' for r in report['runs']) else 'incomplete'
except Exception as exc:report['status']='failed';report['error_type']=type(exc).__name__
finally:
 report['rate_limit']=limiter.summary();save_json(output/'summary.json',report);print('FINAL_EVALUATION_REPORT='+str(output/'summary.json'),flush=True)
