import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock
import numpy as np

from src.config import Settings
from src.retrieval.gemini import GeminiEmbedder
from src.llm import LLMError


class GeminiEmbeddingTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.settings=Settings.load({'RUNTIME_DIR':self.temp.name,'EMBEDDING_API_KEY':'embedding-secret'})
        self.post=Mock(side_effect=self.response)
        self.embedder=GeminiEmbedder(self.settings,post=self.post,sleep=Mock())

    def response(self,*args,**kwargs):
        response=Mock(status_code=200)
        response.json.return_value={'embeddings':[{'values':[1.0]*768} for _ in kwargs['json']['requests']]}
        return response

    def test_154_chunks_batch_into_two_requests_and_cache_across_instances(self):
        texts=[f'public policy {i}' for i in range(154)]
        vectors=self.embedder.passages(texts)
        self.assertEqual(vectors.shape,(154,768))
        np.testing.assert_allclose(np.linalg.norm(vectors,axis=1),1,atol=1e-6)
        self.assertEqual(self.post.call_count,2)
        self.assertEqual([len(c.kwargs['json']['requests']) for c in self.post.call_args_list],[100,54])
        fresh=GeminiEmbedder(self.settings,post=self.post)
        fresh.passages(texts)
        self.assertEqual(self.post.call_count,2)
        self.assertNotIn('embedding-secret',(Path(self.temp.name)/'gemini_embeddings/vectors.sqlite3').read_bytes().decode('latin1'))

    def test_prefetch_52_queries_uses_one_request_and_search_cache(self):
        texts=[f'query {i}' for i in range(52)]
        self.embedder.prefetch_queries(texts)
        for text in texts:self.embedder.queries([text])
        self.assertEqual(self.post.call_count,1)
        request=self.post.call_args.kwargs
        self.assertEqual(request['headers']['x-goog-api-key'],'embedding-secret')
        self.assertNotIn('embedding-secret',self.post.call_args.args[0])
        self.assertEqual(request['json']['requests'][0]['taskType'],'RETRIEVAL_QUERY')
        for path in (Path(self.temp.name)/'gemini_embeddings').glob('*'):
            self.assertNotIn(b'query 1',path.read_bytes())

    def test_query_and_document_vectors_have_separate_cache_keys(self):
        self.embedder.passages(['same text']);self.embedder.queries(['same text'])
        self.assertEqual(self.post.call_count,2)

    def test_invalid_vectors_and_provider_errors_do_not_fall_back(self):
        response=Mock(status_code=200);response.json.return_value={'embeddings':[{'values':[0.0]*768}]}
        self.post.side_effect=None;self.post.return_value=response
        with self.assertRaises(LLMError):self.embedder.queries(['zero'])
        response.status_code=401
        with self.assertRaises(LLMError) as error:self.embedder.queries(['secret text'])
        self.assertEqual(error.exception.code,'http_401')


class BatchJudgeTests(unittest.TestCase):
    def test_ten_answers_share_one_judge_request(self):
        from src.evaluation.judge import grade_run_batched
        from src.evaluation.cases import load_questions
        from src.config import PROJECT_ROOT
        from src.schemas import GenerationTrace
        cases=load_questions(PROJECT_ROOT/'Dataset')[:10]
        with TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'manifest.json').write_text(json.dumps({'case_ids':[c.case_id for c in cases]}))
            (root/'results.jsonl').write_text(''.join(json.dumps({'case_id':c.case_id,'status':'completed','decision_correct':True,'response':{'reply':'actual saved answer'}})+'\n' for c in cases))
            client=Mock();client.settings=Settings.load({'LLM_MODEL':'test'})
            client.complete.return_value=GenerationTrace(json.dumps([{'case_id':c.case_id,'facts_correct':True,'forbidden_claims_absent':True,'language_correct':True,'ai_disclosed':True,'rationale':'Matches the reference.'} for c in cases]),'test')
            summary=grade_run_batched(root,cases,client,batch_size=10)
            self.assertEqual(client.complete.call_count,1)
            self.assertEqual(summary['correct'],10)
            self.assertEqual(summary['unresolved'],0)


if __name__=='__main__':unittest.main()
