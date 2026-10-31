"""Check the local PS submission without API calls or publishing anything.

Needs the application's dependencies, ffprobe (ffmpeg) and pdfinfo (Poppler).
"""
from pathlib import Path
import hashlib
import json
import re
import subprocess
import tempfile
import zipfile
from dotenv import dotenv_values

ROOT=Path(__file__).resolve().parents[1]


def verify():
    required={'README.md','.env.example','requirements.txt','run.py','report.pdf','demo.mp4','PS_COMPLETION.md'}
    keys=[str(v).encode() for k,v in dotenv_values(ROOT/'.env').items() if v and 'KEY' in k.upper() and len(str(v))>=8]
    with zipfile.ZipFile(ROOT/'zentromart_voiceops.zip') as archive:
        assert archive.testzip() is None,'Corrupt ZIP'
        names=archive.namelist();prefix='zentromart_voiceops/'
        assert all(n.startswith(prefix) and '..' not in Path(n).parts for n in names),'Unsafe archive path'
        relative={n[len(prefix):] for n in names}
        assert required<=relative,'Required PS deliverable missing'
        assert any(n.startswith('src/') for n in relative) and any(n.startswith('sample_outputs/') for n in relative)
        for name in relative:
            parts=Path(name).parts
            assert not any(p in {'.git','.venv','Dataset','runtime','__pycache__'} for p in parts),'Private/generated path in ZIP'
            assert Path(name).name!='.env','Secret configuration in ZIP'
            assert not any(key in archive.read(prefix+name) for key in keys),'Configured credential in ZIP'
        manifest=json.loads(archive.read(prefix+'release_manifest.json'))
        assert set(manifest)==relative-{'release_manifest.json'}
        assert all(hashlib.sha256(archive.read(prefix+n)).hexdigest()==h for n,h in manifest.items()),'File hash mismatch'
        example=dotenv_values(stream=__import__('io').StringIO(archive.read(prefix+'.env.example').decode()))
        assert all(not v for k,v in example.items() if 'KEY' in k.upper()),'Example credentials must be blank'
        readme=archive.read(prefix+'README.md').decode()
        assert all(flag in readme for flag in ('--input-dir','--output-dir','--dataset-dir'))
        assert 'https://github.com/TejasKS07/ZentroMart-VoiceOps' in readme,'Repository URL missing'
        with tempfile.TemporaryDirectory(prefix='voiceops-verify-') as folder:
            folder=Path(folder);pdf=folder/'report.pdf';video=folder/'demo.mp4'
            pdf.write_bytes(archive.read(prefix+'report.pdf'));video.write_bytes(archive.read(prefix+'demo.mp4'))
            info=subprocess.run(['pdfinfo',str(pdf)],capture_output=True,text=True,check=True).stdout
            pages=int(re.search(r'^Pages:\s*(\d+)',info,re.M)[1]);assert 1<=pages<=4,'Report exceeds four pages'
            probe=json.loads(subprocess.run(['ffprobe','-v','error','-show_entries','format=duration:stream=codec_type','-of','json',str(video)],capture_output=True,text=True,check=True).stdout)
            duration=float(probe['format']['duration']);assert 0<duration<=120,'Demo exceeds two minutes'
            assert {s['codec_type'] for s in probe['streams']}>={'audio','video'},'Demo needs picture and sound'
    return {'status':'passed','archive_files':len(relative),'verified_file_hashes':len(manifest),'report_pages':pages,
        'demo_seconds':duration,'credential_scan':'passed','dataset_and_private_runtime':'excluded',
        'api_calls':0,'publication':'kept local by user request'}


if __name__=='__main__':print(json.dumps(verify(),indent=2))
