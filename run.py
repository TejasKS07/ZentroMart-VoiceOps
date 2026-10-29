#!/usr/bin/env python3
"""Submission interface: audio files -> grounded text, Gnani speech and evidence."""

import argparse
import json
from pathlib import Path
import sys

from src.config import Settings
from src.factory import create_agent
from src.speech.provider import GnaniSpeech
from src.speech.engine import VoiceEngine
from src.batch import run_batch


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--dataset-dir', type=Path)
    parser.add_argument('--manifest', type=Path, help='Optional ordered conversation JSON manifest')
    parser.add_argument('--language', choices=('en-IN','hi-IN','hi-en','indic-auto'), default='en-IN')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--offline', action='store_true', help='Local policy fallback; Gnani speech still requires the configured key')
    args = parser.parse_args(argv)
    try:
        settings = Settings.load()
        dataset = args.dataset_dir.resolve() if args.dataset_dir else settings.dataset_dir
        if args.output_dir.resolve().is_relative_to(dataset):
            raise ValueError('Output must be outside the dataset directory')
        agent = create_agent(settings, dataset_dir=dataset, live=not args.offline,
                             tool_dir=settings.runtime_dir / 'batch_tools')
        engine = VoiceEngine(agent, GnaniSpeech(settings))
        summary = run_batch(engine, args.input_dir, args.output_dir, manifest=args.manifest,
                            language=args.language, resume=args.resume, progress=lambda text: print(text,file=sys.stderr,flush=True))
        print(json.dumps(summary,indent=2))
        return 0 if summary['status']=='completed' else 2
    except Exception as exc:
        # Configuration and provider internals never become raw submission logs.
        print(json.dumps({'status':'failed','error_type':type(exc).__name__, 'code':getattr(exc,'code','configuration_or_input_error')}),file=sys.stderr)
        return 1


if __name__=='__main__':
    raise SystemExit(main())
