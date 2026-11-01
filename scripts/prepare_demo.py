"""Synthetic caller audio and actual agent replies, explicitly not human recordings."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.config import Settings
from src.factory import create_agent
from src.speech.provider import GnaniSpeech
from src.speech.engine import VoiceEngine
from src.evaluation.runner import save_json
settings=Settings.load();speech=GnaniSpeech(settings)
agent=create_agent(settings,tool_dir=settings.runtime_dir/'demo_tools');engine=VoiceEngine(agent,speech)
inputs=settings.runtime_dir/'voice_inputs'/'demo';outputs=settings.runtime_dir/'demo'
inputs.mkdir(parents=True,exist_ok=True);outputs.mkdir(exist_ok=True)
records=[]
for i,question in enumerate(('How long does a UPI refund take?','And what about the wallet?','I would like a human agent.'),1):
 path=inputs/f'{i:02d}.wav'
 if not path.exists():path.write_bytes(speech.synthesize(question,'en'))
 result=engine.process(path,'synthetic-demo-final','en-IN')
 if result.audio:(outputs/f'{i:02d}-reply.wav').write_bytes(result.audio)
 if result.response:(outputs/f'{i:02d}-reply.txt').write_text(result.response.reply)
 save_json(outputs/f'{i:02d}-result.json',result.to_dict())
 records.append({'turn':i,'input':str(path),'result':result.to_dict()})
 print(i,result.status,result.response.reply if result.response else result.error,flush=True)
save_json(outputs/'manifest.json',{'kind':'synthetic_loopback_demo','human_recording':False,'records':records})
