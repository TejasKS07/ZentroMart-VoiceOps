"""Build an honest playback demo from saved Gnani inputs and real agent outputs."""
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
import json,subprocess,wave,shutil
ROOT=Path(__file__).resolve().parents[1]
folder=ROOT/'runtime/demo';manifest=json.loads((folder/'manifest.json').read_text())
font_path='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
regular=lambda size:ImageFont.truetype(font_path,size)
segments=[]
def wrap(text,font,width,draw):
 lines=[];line=''
 for word in text.split():
  trial=(line+' '+word).strip()
  if draw.textlength(trial,font=font)>width and line:lines.append(line);line=word
  else:line=trial
 if line:lines.append(line)
 return lines
def frame(number,question,answer,sources,active):
 image=Image.new('RGB',(1280,720),'#f3f5ef');draw=ImageDraw.Draw(image)
 draw.rounded_rectangle((40,30,84,74),12,fill='#254f40');draw.text((51,35),'Z',font=regular(27),fill='white')
 draw.text((102,39),'ZentroMart VoiceOps',font=regular(22),fill='#254f40')
 draw.text((875,44),'SYNTHETIC LOOPBACK DEMO',font=regular(15),fill='#72816b')
 draw.text((60,105),f'{number:02d}  '+('Document-grounded answer' if number==1 else 'Contextual follow-up' if number==2 else 'Human handover'),font=regular(30),fill='#263d2e')
 draw.rounded_rectangle((60,168,1220,310),16,fill='#e6edde',outline='#91aa7a' if active=='caller' else '#d8e1cf',width=2)
 draw.text((85,187),'CALLER  /  SYNTHETIC GNANI SPEECH',font=regular(13),fill='#75876a')
 y=219
 for line in wrap(question,regular(25),1090,draw):draw.text((85,y),line,font=regular(25),fill='#4c6242');y+=35
 draw.rounded_rectangle((60,334,1220,602),16,fill='white',outline='#698c51' if active=='assistant' else '#dfe6d8',width=2)
 draw.text((85,355),'ASSISTANT  /  ACTUAL SAVED AGENT REPLY',font=regular(13),fill='#75876a')
 y=391
 for line in wrap(answer,regular(25),1080,draw):draw.text((85,y),line,font=regular(25),fill='#3f5636');y+=36
 draw.text((85,562),'Sources: '+sources,font=regular(14),fill='#7f9273')
 draw.text((60,644),'Gnani STT → shared conversation controller → Gnani TTS',font=regular(18),fill='#657c56')
 draw.text((60,681),'Artifact playback; no human microphone recording. Replies were not edited by hand.',font=regular(13),fill='#8a9880')
 path=folder/f'{number:02d}-{active}.png';image.save(path);return path
for record in manifest['records']:
 number=record['turn'];result=record['result'];response=result['response']
 sources=', '.join(response['evidence_ids'])
 if response['tool_events']:sources+=' | '+', '.join(e['name']+': '+e['status'] for e in response['tool_events'])
 for role,audio in [('caller',Path(record['input'])),('assistant',folder/f'{number:02d}-reply.wav')]:
  picture=frame(number,result['transcript'],response['reply'],sources,role)
  with wave.open(str(audio),'rb') as stream:duration=stream.getnframes()/stream.getframerate()
  segment=folder/f'{number:02d}-{role}.mp4'
  subprocess.run(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-y','-loop','1','-framerate','24','-i',str(picture),'-i',str(audio),'-c:v','libx264','-preset','fast','-tune','stillimage','-pix_fmt','yuv420p','-c:a','aac','-b:a','128k','-af','apad=pad_dur=0.6','-t',str(duration+0.6),'-movflags','+faststart',str(segment)],check=True)
  segments.append(segment)
listing=folder/'concat.txt';listing.write_text(''.join("file '"+str(p)+"'\n" for p in segments))
output=ROOT/'output';output.mkdir(exist_ok=True)
subprocess.run(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-y','-f','concat','-safe','0','-i',str(listing),'-c','copy','-movflags','+faststart',str(output/'demo.mp4')],check=True)
probe=subprocess.run(['ffprobe','-v','error','-show_entries','format=duration','-of','json',str(output/'demo.mp4')],capture_output=True,text=True,check=True)
seconds=float(json.loads(probe.stdout)['format']['duration']);assert seconds<=120
(output/'demo_metadata.json').write_text(json.dumps({'duration_seconds':seconds,'kind':'synthetic_loopback_artifact_playback','manually_edited_answers':False},indent=2))
print(output/'demo.mp4',seconds)
