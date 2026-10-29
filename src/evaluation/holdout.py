"""Small authored holdout, frozen before the final generation pass."""

from pathlib import Path
from uuid import uuid4
from .cases import EvaluationCase
from .runner import run_questions
from .judge import grade_run
from ..schemas import CustomerInput

# New phrasings derived from current KB rules; never supplied to the agent.
ROWS = (
 ('H01','Can two different discount codes be combined at checkout?', 'Only one coupon per order.', 'one coupon / single coupon', 'CPN-1'),
 ('H02','If I change my mind about fresh vegetables, can I send them back?', 'Grocery is not returnable for change of mind.', 'not returnable / cannot return', 'RET-5'),
 ('H03','After a refund starts, what is the maximum wait for Wallet credit?', 'Within two hours after initiation.', '2 hours / two hours', 'REF-2'),
 ('H04','Do Plus customers get a late Express fee credit when they paid no delivery fee?', 'No. Plus customers paid no Express fee and get no credit.', 'no credit / no refund', 'DEL-4'),
 ('H05','My parcel is packed but has not shipped; can you cancel it for me?', 'The customer can cancel free in My Orders before shipping; the assistant cannot perform cancellation.', 'My Orders ; cannot cancel / cannot do it / guide', 'CAN-1'),
 ('H06','A sealed beauty item was delivered yesterday. What is the standard return period?', 'Seven calendar days from delivery, unopened and seal intact.', '7 days / seven days ; unopened / seal', 'RET-2'),
 ('H07','What evidence is required when an item arrives broken?', 'Report within 48 hours with photos of the item and packaging.', '48 hours / forty-eight hours ; photos ; packaging', 'DWM-1'),
 ('H08','Is there a documented policy for shipping my order to the Moon?', 'The supplied policies do not offer Moon delivery; delivery is within India only.', 'India / not available / no information', 'DEL-5'),
)


def cases():
    return tuple(EvaluationCase(i,CustomerInput(q,i),'en',1,('answer','clarify','refuse'),(source,),
        {'reference_answer':answer,'must_include':required,'must_not_include':'','reply_language':'en','question':q,'answerable':'True'})
        for i,q,answer,required,source in ROWS)


def evaluate(settings, index, output_dir: Path, *, judge=False, progress=None):
    from ..llm import ChatCompletionClient
    from ..rag import RagAgent
    client=ChatCompletionClient(settings)
    secrets=(settings.llm_api_key,settings.gnani_api_key)
    summary=run_questions(cases(),RagAgent(client,index).respond,output_dir,secrets=secrets,
        metadata={'stage':'authored_holdout','frozen_before_generation':True,'requested_model':settings.llm_model,'generation_parameters':settings.generation_parameters()},progress=progress)
    if judge:summary['grading']=grade_run(output_dir,cases(),client,secrets=secrets,progress=progress)
    return summary
