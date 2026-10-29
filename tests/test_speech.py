"""Audio boundaries, conservative identifiers, privacy and honest voice failure."""

import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock
import wave

import numpy as np

from src.config import PROJECT_ROOT, Settings
from src.conversation import ConversationAgent
from src.tools.operations import SupportOperations
from src.speech.audio import SpeechError, inspect_wav, normalize_audio
from src.speech.provider import GnaniSpeech
from src.speech.engine import VoiceEngine
from src.speech.text import normalize_identifiers, spoken_reply
from src.batch import audio_cases, run_batch


def wav(seconds=0.3, *, silent=False, rate=16000):
    data=io.BytesIO()
    with wave.open(data,'wb') as stream:
        stream.setnchannels(1);stream.setsampwidth(2);stream.setframerate(rate)
        signal=np.zeros(int(seconds*rate)) if silent else np.sin(np.arange(int(seconds*rate))*0.12)*5000
        stream.writeframes(signal.astype('<i2').tobytes())
    return data.getvalue()


class SpeechTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.input=self.root/'in';self.input.mkdir()
        self.clip=self.input/'sample.wav';self.clip.write_bytes(wav())
        self.agent=ConversationAgent(SupportOperations(PROJECT_ROOT/'Dataset',self.root/'tools'))
        self.speech=Mock()
        self.speech.settings=Settings.load({'GNANI_API_KEY':'test-key'})
        self.speech.transcribe.return_value='Can I use two coupons?'
        self.speech.synthesize.return_value=wav()
        self.engine=VoiceEngine(self.agent,self.speech)

    def test_audio_normalization_and_silent_corrupt_duration_boundaries(self):
        self.assertEqual(inspect_wav(normalize_audio(self.clip)[0]).sample_rate,16000)
        for data,code in ((wav(silent=True),'silent_audio'),(b'garbage','decode_failed'),(wav(31),'invalid_audio')):
            self.clip.write_bytes(data)
            with self.assertRaises(SpeechError) as caught:normalize_audio(self.clip)
            self.assertEqual(caught.exception.code,code)
        self.clip.write_bytes(wav(rate=48000))
        self.assertEqual(normalize_audio(self.clip)[1].sample_rate,16000)

    def test_spoken_identifiers_do_not_guess_homophones_or_modify_amount_questions(self):
        text,ambiguous=normalize_identifiers('Order number eight nine seven eight two eight three seven')
        self.assertIn('89782837',text);self.assertFalse(ambiguous)
        self.assertEqual(normalize_identifiers('ऑर्डर ८९७८ २८३७')[0],'ऑर्डर 8978 2837')
        self.assertTrue(normalize_identifiers('order eight nine seven to eight three seven')[1])
        self.assertEqual(normalize_identifiers('one to three business days'),('one to three business days',False))
        self.assertIn('eight nine seven eight, two eight three seven',spoken_reply('Order 89782837'))
        self.assertIn('49 rupees',spoken_reply('The fee is ₹49.'))

    def test_provider_uses_explicit_settings_safe_retries_and_hinglish_auto(self):
        import requests
        stt,tts=Mock(),Mock()
        stt.transcribe_bytes.side_effect=[requests.Timeout('private body'),{'transcript':'clear question'}]
        tts.synthesize.return_value=wav()
        sleep=Mock()
        adapter=GnaniSpeech(self.speech.settings,stt=stt,tts=tts,sleep=sleep)
        self.assertEqual(adapter.transcribe(wav(),'hi-en'),'clear question')
        self.assertEqual(stt.transcribe_bytes.call_count,2)
        self.assertEqual(stt.transcribe_bytes.call_args.kwargs['model'],'prisma-v3.0')
        adapter.synthesize('Ek coupon','hi-en')
        self.assertEqual(tts.synthesize.call_args.kwargs['language'],'auto')
        streaming=bytearray(wav());streaming[40:44]=(0xfffffffe).to_bytes(4,'little')
        tts.synthesize.return_value=bytes(streaming)
        self.assertLess(inspect_wav(adapter.synthesize('hello')).duration_seconds,1)
        stt.transcribe_bytes.side_effect=requests.Timeout('private secret')
        with self.assertRaises(SpeechError) as error:adapter.transcribe(wav())
        self.assertNotIn('private secret',str(error.exception))

    def test_incomplete_transcript_and_invalid_synthesis_are_failures(self):
        stt,tts=Mock(),Mock()
        adapter=GnaniSpeech(self.speech.settings,stt=stt,tts=tts,sleep=Mock())
        for response in ({'transcript':''},{'transcript':'hello','truncated':True},{'transcript':'---'}):
            stt.transcribe_bytes.return_value=response
            with self.assertRaises(SpeechError):adapter.transcribe(wav())
        tts.synthesize.return_value=b'not audio'
        with self.assertRaises(SpeechError):adapter.synthesize('hello')

    def test_same_core_evidence_and_timings_reach_voice_output(self):
        result=self.engine.process(self.clip,'one')
        self.assertEqual(result.status,'completed')
        self.assertTrue(result.to_dict()['voice_complete'])
        self.assertIn('CPN-1',[e.evidence_id for e in result.response.evidence])
        self.assertTrue({'stt','tts','agent','total'}<=set(result.timings_ms))

    def test_verification_words_are_redacted_and_tts_retry_does_not_verify_again(self):
        self.agent.respond(__import__('src.schemas',fromlist=['CustomerInput']).CustomerInput('order 18599253','one'))
        self.speech.transcribe.return_value='four two six five'
        self.speech.synthesize.side_effect=SpeechError('tts_failed','tts','Speech failed')
        result=self.engine.process(self.clip,'one')
        self.assertTrue(self.agent.snapshot('one').verified)
        self.assertNotIn('4265',json.dumps(result.to_dict()))
        self.assertNotIn('four two six five',json.dumps(result.to_dict()))
        self.assertEqual(result.status,'failed')
        self.speech.synthesize.side_effect=None
        self.engine.synthesize(result)
        self.assertEqual(result.status,'completed')
        self.assertEqual(self.speech.transcribe.call_count,1)

    def test_ambiguous_identifier_and_silence_never_call_order_tools(self):
        self.speech.transcribe.return_value='order eight nine seven to three four five'
        result=self.engine.process(self.clip,'one')
        self.assertEqual(result.status,'needs_repeat')
        self.assertFalse(self.agent.snapshot('one').verified)
        self.assertEqual(result.response.tool_events,())
        self.clip.write_bytes(wav(silent=True))
        self.speech.transcribe.reset_mock()
        result=self.engine.process(self.clip,'two')
        self.speech.transcribe.assert_not_called()
        self.assertEqual(result.status,'needs_repeat')

    def test_batch_writes_actual_artifacts_and_skips_complete_resume(self):
        output=self.root/'out'
        summary=run_batch(self.engine,self.input,output)
        self.assertEqual(summary['voice_completed'],1)
        self.assertTrue((output/'clip-0001/reply.wav').is_file())
        self.assertIn('CPN-1',(output/'clip-0001/evidence.json').read_text())
        count=self.speech.transcribe.call_count
        run_batch(self.engine,self.input,output,resume=True)
        self.assertEqual(self.speech.transcribe.call_count,count)
        self.clip.write_bytes(wav(0.4))
        with self.assertRaises(ValueError):run_batch(self.engine,self.input,output,resume=True)

    def test_mixed_batch_preserves_failure_and_continues(self):
        (self.input/'corrupt.wav').write_bytes(b'not wav')
        summary=run_batch(self.engine,self.input,self.root/'mixed')
        self.assertEqual(summary['clips'],2)
        self.assertEqual(summary['voice_completed'],1)
        self.assertEqual(summary['needs_repeat'],1)
        self.assertEqual(summary['status'],'partial_failure')

    def test_resume_retries_failed_tts_without_replaying_the_order_turn(self):
        self.speech.synthesize.side_effect=SpeechError('tts_failed','tts','Retry speech')
        output=self.root/'retry'
        self.assertEqual(run_batch(self.engine,self.input,output)['failed'],1)
        count=self.speech.transcribe.call_count
        self.speech.synthesize.side_effect=None
        self.assertEqual(run_batch(self.engine,self.input,output,resume=True)['voice_completed'],1)
        self.assertEqual(self.speech.transcribe.call_count,count)

    def test_manifest_rejects_escape_duplicate_and_missing_turns(self):
        manifest=self.root/'manifest.json'
        for rows in ([{'conversation_id':'one','turn':1,'file':'../outside.wav','language':'en-IN'}],
                     [{'conversation_id':'one','turn':2,'file':'sample.wav','language':'en-IN'}]):
            manifest.write_text(json.dumps(rows))
            with self.assertRaises(ValueError):audio_cases(self.input,manifest=manifest)


if __name__=='__main__':unittest.main()
