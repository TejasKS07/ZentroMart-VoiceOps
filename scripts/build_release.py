"""Package an explicit source/artifact allowlist; never include runtime or Dataset."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    from dotenv import dotenv_values
    secrets = [str(v).encode() for k, v in dotenv_values(ROOT / '.env').items()
               if v and 'KEY' in k.upper() and len(str(v)) >= 8]
    files = [ROOT / name for name in (
        'README.md', 'PLAN_OF_ACTION.md', 'PHASE_2_3_REPORT.md', 'PHASE_4_REPORT.md',
        'PHASE_5_7_REPORT.md', '.env.example', '.gitignore', '.gitattributes', 'requirements.txt',
        'run.py', 'report.pdf', 'demo.mp4')]
    for directory in ('src', 'web', 'tests', 'scripts', 'sample_outputs'):
        files.extend(p for p in (ROOT / directory).rglob('*') if p.is_file()
                     and '__pycache__' not in p.parts and p.suffix not in {'.pyc', '.pyo'})
    assert all(p.is_file() for p in files), 'Required release artifact is missing'
    names = [str(p.relative_to(ROOT)) for p in sorted(set(files))]
    assert not any(n.startswith(('Dataset/', 'runtime/', '.venv/', '.git/')) for n in names)
    manifest = {}
    for name in names:
        data = (ROOT / name).read_bytes()
        assert not any(secret in data for secret in secrets), 'Credential scan failed: ' + name
        assert Path(name).name != '.env'
        manifest[name] = hashlib.sha256(data).hexdigest()
    output = ROOT / 'zentromart_voiceops.zip'
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            archive.write(ROOT / name, 'zentromart_voiceops/' + name)
        archive.writestr('zentromart_voiceops/release_manifest.json', json.dumps(manifest, indent=2))
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == len(names) + 1
    print(json.dumps({'zip': str(output), 'files': len(names), 'bytes': output.stat().st_size,
                      'credential_scan': 'passed', 'excluded': ['Dataset', '.env', 'runtime', '.venv', '.git']}))


if __name__ == '__main__':
    main()
