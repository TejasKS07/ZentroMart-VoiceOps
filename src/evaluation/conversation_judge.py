"""Semantic review of saved conversation replies; never used for generation."""

import json
import re
from pathlib import Path
from .cases import load_conversations
from .judge import CHECKS, JUDGE_PROMPT
from .runner import save_json
from ..agent import parse_object
from ..redaction import redact, normalize_digits

PROMPT=JUDGE_PROMPT.replace("agent's first reply","agent reply in a conversation").replace(
 'whether this first\nreply clearly discloses being an AI assistant',
 'whether the first reply clearly discloses being an AI assistant').replace(
 'Return only JSON with exactly these keys:',
 'The is_first_reply field tells you whether AI disclosure is required on this turn. For later turns, set ai_disclosed=true without requiring a repeated introduction. Use the supplied prior replies to interpret follow-ups. Return only JSON with exactly these keys:')


def grade_conversations(run_dir: Path, dataset_dir: Path, client, *, secrets=(), progress=None):
    cases={c.case_id:c for group in load_conversations(dataset_dir).values() for c in group}
    rows=[json.loads(line) for line in (run_dir/'results.jsonl').read_text().splitlines() if line]
    proofs=[]
    for case in cases.values():
        if re.search(r'last four|digits|pincode|अंक|पिनकोड',case.customer.text,re.I):
            proofs.extend(re.findall(r'(?<!\d)(?:\d[ -]*){4,6}(?!\d)',normalize_digits(case.customer.text)))
    private=tuple(secrets)+tuple(re.sub(r'\D','',p) for p in proofs)
    scores=[];previous={}
    with (run_dir/'semantic_judgements.jsonl').open('x',encoding='utf-8') as handle:
        for row in rows:
            case=cases[row['case_id']]
            group=row['conversation_id']
            score={'case_id':case.case_id,'decision_correct':row.get('decision_correct',False)}
            try:
                if row['status']!='completed':raise ValueError('Failed generation')
                payload={'question':case.customer.text,'candidate_reply':row['response']['reply'],
                    'reference_answer':case.labels.get('notes',''),'must_include':case.labels.get('must_include',''),
                    'must_not_include':case.labels.get('must_not_include',''),'reply_language':case.language,
                    'is_first_reply':case.turn==1,'prior_replies':previous.get(group,[])[-3:]}
                trace=client.complete([{'role':'system','content':PROMPT},{'role':'user','content':json.dumps(redact(payload,private),ensure_ascii=False)}])
                result=parse_object(trace.raw_output)
                if set(result)!={*CHECKS,'rationale'} or any(type(result[k]) is not bool for k in CHECKS):raise ValueError('Invalid judge')
                score.update(status='graded',passed=bool(score['decision_correct'] and all(result[k] for k in CHECKS)),checks=result,generation=trace.to_dict(private))
            except Exception as exc:
                score.update(status='unresolved',passed=None,error_type=type(exc).__name__)
            scores.append(score);handle.write(json.dumps(redact(score,private),ensure_ascii=False)+'\n');handle.flush()
            if row.get('response'):previous.setdefault(group,[]).append(row['response']['reply'])
            if progress:progress(f"Conversation grading {len(scores)}/{len(rows)}")
    summary={'method':'provisional_model_judge','turns':len(rows),'correct':sum(s['passed'] is True for s in scores),
             'unresolved':sum(s['passed'] is None for s in scores),'human_review':'not performed'}
    summary['accuracy']=summary['correct']/len(rows) if rows and not summary['unresolved'] else None
    save_json(run_dir/'semantic_summary.json',summary)
    return summary
