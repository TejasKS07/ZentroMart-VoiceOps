"""Gemini Embedding 1 with synchronous batching and a hash-only vector cache.

Only public KB/catalog passages and sanitized retrieval queries belong here.
No raw query text, credentials or provider response bodies are persisted.
"""
import hashlib
import json
from pathlib import Path
import sqlite3
from threading import RLock
import time

import numpy as np
from ..config import ConfigurationError
from ..llm import LLMError


class GeminiEmbedder:
    model='gemini-embedding-001'
    dimensions=768
    name='google/gemini-embedding-001-768'
    fingerprint=hashlib.sha256(b'gemini-embedding-001/768/RETRIEVAL_DOCUMENT/RETRIEVAL_QUERY/v1').hexdigest()
    batch_size=100

    def __init__(self, settings, *, post=None, sleep=time.sleep, request_limiter=None):
        if not settings.embedding_api_key:
            raise ConfigurationError('Set EMBEDDING_API_KEY or the Google LLM_API_KEY for Gemini embeddings')
        self.settings=settings
        self.cache_dir=settings.runtime_dir/'gemini_embeddings'
        self.cache_dir.mkdir(parents=True,exist_ok=True)
        self.path=self.cache_dir/'vectors.sqlite3'
        self.lock=RLock();self.network_requests=0;self.sleep=sleep
        self.request_limiter=request_limiter
        if post is None:
            import requests
            post=requests.post
        self.post=post
        with sqlite3.connect(self.path) as connection:
            connection.execute('CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, value BLOB NOT NULL)')

    def _key(self, text, task):
        return hashlib.sha256((self.fingerprint+'\n'+task+'\n'+text).encode()).hexdigest()

    def _request(self, texts, task):
        import requests
        body={'requests':[{'model':'models/'+self.model,'content':{'parts':[{'text':text}]},
            'taskType':task,'outputDimensionality':self.dimensions} for text in texts]}
        for attempt in range(2):
            lease=self.request_limiter.acquire(len(texts)) if self.request_limiter else None
            self.network_requests+=1
            try:
                response=self.post('https://generativelanguage.googleapis.com/v1beta/models/'+self.model+':batchEmbedContents',
                    headers={'x-goog-api-key':self.settings.embedding_api_key,'Content-Type':'application/json'},
                    json=body,timeout=self.settings.timeout_seconds,allow_redirects=False)
            except requests.RequestException:
                if self.request_limiter:self.request_limiter.record(started=lease,kind='embedding',weight=len(texts),status='connection_failed')
                if attempt==0:self.sleep(1);continue
                raise LLMError('Gemini embedding connection failed',code='connection_failed') from None
            if self.request_limiter:self.request_limiter.record(started=lease,kind='embedding',weight=len(texts),status=response.status_code)
            if response.status_code in {408,429,500,502,503,504} and attempt==0:
                delay=1
                if response.status_code==429:
                    from ..rate_limit import retry_delay
                    delay=retry_delay(response)
                if self.request_limiter:self.request_limiter.cooldown(delay)
                else:self.sleep(delay)
                continue
            if response.status_code!=200:
                raise LLMError(f'Gemini embeddings returned HTTP {response.status_code}',code=f'http_{response.status_code}')
            try:
                payload=response.json()
                values=np.asarray([item['values'] for item in payload['embeddings']],dtype=np.float32)
                if values.shape!=(len(texts),self.dimensions) or not np.isfinite(values).all():raise ValueError
                norms=np.linalg.norm(values,axis=1,keepdims=True)
                if np.any(norms==0):raise ValueError
                values=values/norms
            except (ValueError,TypeError,KeyError):
                raise LLMError('Gemini returned invalid embedding vectors',code='invalid_embeddings') from None
            with (self.cache_dir/'requests.jsonl').open('a',encoding='utf-8') as handle:
                handle.write(json.dumps({'model':self.model,'task':task,'items':len(texts),'dimensions':self.dimensions})+'\n')
            return values

    def _embed(self, texts, task):
        texts=list(texts)
        if not texts:return np.empty((0,self.dimensions),dtype=np.float32)
        if any(not isinstance(text,str) or not text.strip() for text in texts):raise ValueError('Embedding text must not be empty')
        keys=[self._key(text,task) for text in texts]
        with self.lock,sqlite3.connect(self.path) as connection:
            cached={}
            for key in dict.fromkeys(keys):
                row=connection.execute('SELECT value FROM vectors WHERE key=?',(key,)).fetchone()
                if row:
                    vector=np.frombuffer(row[0],dtype='<f4')
                    if vector.shape==(self.dimensions,) and np.isfinite(vector).all():cached[key]=vector.copy()
            missing=list(dict.fromkeys(key for key in keys if key not in cached))
            lookup=dict(zip(keys,texts))
            batch_size=min(self.batch_size,self.request_limiter.capacity) if self.request_limiter else self.batch_size
            for start in range(0,len(missing),batch_size):
                group=missing[start:start+batch_size]
                vectors=self._request([lookup[key] for key in group],task)
                for key,vector in zip(group,vectors):
                    cached[key]=vector
                    connection.execute('INSERT OR REPLACE INTO vectors VALUES (?,?)',(key,vector.astype('<f4').tobytes()))
                connection.commit()
            return np.asarray([cached[key] for key in keys],dtype=np.float32)

    def passages(self,texts):return self._embed(texts,'RETRIEVAL_DOCUMENT')
    def queries(self,texts):return self._embed(texts,'RETRIEVAL_QUERY')
    def prefetch_queries(self,texts):return self.queries([text.strip() for text in texts])
