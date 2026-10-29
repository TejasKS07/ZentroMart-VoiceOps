"""Copy completed experiment evidence without modifying saved answers."""
from pathlib import Path
import json
import shutil

ROOT=Path(__file__).resolve().parents[1]
sample=ROOT/'sample_outputs'
report_path=Path(json.loads((ROOT/'runtime/final_evaluation/latest.json').read_text())['report'])
report=json.loads(report_path.read_text())
assert report['status']=='finished'
evaluation=sample/'evaluation';evaluation.mkdir(exist_ok=True)
shutil.copy2(report_path,evaluation/'summary.json')
shutil.copy2(report_path.parent/'requests.jsonl',evaluation/'requests.jsonl')
index=[]
for run in report['runs']:
    folder=evaluation/(run['stage']+'_'+run['variant']);folder.mkdir(exist_ok=True)
    for name in ('manifest.json','results.jsonl','summary.json','judgements.jsonl',
                 'judge_manifest.json','judge_batches.jsonl','grading_summary.json','report.md'):
        p=Path(run['output_dir'])/name
        if p.exists():shutil.copy2(p,folder/name)
    index.append({'stage':run['stage'],'variant':run['variant'],'directory':str(folder.relative_to(sample))})
(evaluation/'release_index.json').write_text(json.dumps(index,indent=2)+'\n')
analysis=Path(json.loads((ROOT/'runtime/release_analysis/latest.json').read_text())['report']).parent
shutil.copytree(analysis,sample/'analysis',dirs_exist_ok=True)
print('Copied completed evaluation and zero-network analysis into sample_outputs')
