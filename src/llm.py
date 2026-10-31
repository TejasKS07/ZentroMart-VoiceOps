"""Text-only Chat Completions transport shared by smoke, agent and evaluator.

Retries transient transport failures, never repairs/re-generates model answers.
No audio input, web search, provider tools, policy files or evaluation labels.
"""

import time
from typing import Any
from urllib.parse import urlsplit

from .config import ConfigurationError, Settings
from .schemas import GenerationTrace


class LLMError(RuntimeError):
    def __init__(self, message: str, *, code: str = "llm_error"):
        super().__init__(message)
        self.code = code


class ChatCompletionClient:
    max_attempts = 3

    def __init__(self, settings: Settings, *, request_limiter=None):
        if not settings.llm_endpoint or not settings.llm_model:
            raise ConfigurationError("Set LLM_ENDPOINT and LLM_MODEL before making LLM requests")
        if not settings.llm_calls_enabled:
            raise ConfigurationError("LLM calls are deferred. Set LLM_CALLS_ENABLED=true only when ready for the final project evaluation.")
        self.settings = settings
        self.request_limiter=request_limiter

    def _request(self,messages,params,response_schema):
        """Gemma 4 uses Google's documented native text endpoint exclusively."""
        google=urlsplit(self.settings.llm_endpoint).hostname=='generativelanguage.googleapis.com'
        native=google and self.settings.llm_model.startswith('gemma-4-')
        if native:
            if self.settings.llm_model not in {'gemma-4-26b-a4b-it','gemma-4-31b-it'}:
                raise ConfigurationError('Choose a documented hosted Gemma 4 model ID')
            systems=[];contents=[]
            for message in messages:
                role=message['role'];content=message['content']
                if not isinstance(content,str):raise ValueError('Text input only')
                if role in {'system','developer'}:systems.append({'text':content})
                elif role in {'user','assistant'}:contents.append({'role':'model' if role=='assistant' else 'user','parts':[{'text':content}]})
                else:raise ValueError('Unsupported text role')
            config={}
            if 'temperature' in params:config['temperature']=params['temperature']
            if 'reasoning_effort' in params:
                effort=params['reasoning_effort']
                if effort not in {'minimal','high'}:raise ConfigurationError('Gemma 4 reasoning effort must be minimal or high')
                config['thinkingConfig']={'thinkingLevel':effort.upper()}
            if response_schema is not None:config.update(responseMimeType='application/json',responseJsonSchema=response_schema)
            body={'contents':contents}
            if systems:body['systemInstruction']={'parts':systems}
            if config:body['generationConfig']=config
            endpoint='https://generativelanguage.googleapis.com/v1beta/models/'+self.settings.llm_model+':generateContent'
            return endpoint,{'Content-Type':'application/json','x-goog-api-key':self.settings.llm_api_key},body,True
        headers={'Content-Type':'application/json'}
        if self.settings.llm_api_key:headers['Authorization']='Bearer '+self.settings.llm_api_key
        body={'model':self.settings.llm_model,'messages':messages,**params}
        if response_schema is not None:
            body['response_format']={'type':'json_schema','json_schema':{'name':'support_reply','strict':True,'schema':response_schema}}
        return self.settings.llm_endpoint,headers,body,False

    def complete(self, messages: list[dict[str, str]], *,
                 parameters: dict[str, Any] | None = None,
                 response_schema: dict | None = None) -> GenerationTrace:
        import requests

        params = self.settings.generation_parameters() if parameters is None else parameters
        if set(params) - {"temperature", "reasoning_effort"}:
            raise ValueError("Unsupported generation parameters")
        endpoint,headers,body,native=self._request(messages,params,response_schema)
        started = time.perf_counter()
        for attempt in range(1, self.max_attempts + 1):
            lease=self.request_limiter.acquire() if self.request_limiter else None
            try:
                response = requests.post(endpoint, headers=headers, json=body,
                                         timeout=self.settings.timeout_seconds, allow_redirects=False)
            except requests.RequestException:
                if self.request_limiter:self.request_limiter.record(started=lease,kind='chat_completion',weight=1,status='connection_failed')
                if attempt == self.max_attempts:
                    raise LLMError("LLM connection failed; check endpoint/network/credentials", code="connection_failed") from None
                time.sleep(attempt)
                continue
            status = response.status_code
            if self.request_limiter:self.request_limiter.record(started=lease,kind='chat_completion',weight=1,status=status)
            if status in {408, 429, 500, 502, 503, 504} and attempt < self.max_attempts:
                if status==429:
                    from .rate_limit import retry_delay
                    delay=retry_delay(response)
                    if self.request_limiter:self.request_limiter.cooldown(delay)
                    else:time.sleep(delay)
                else:time.sleep(attempt)
                continue
            if status != 200:
                raise LLMError(f"LLM request returned HTTP {status}", code=f"http_{status}")
            try:
                data = response.json()
                if native:
                    candidate=data['candidates'][0]
                    text=''.join(part.get('text','') for part in candidate['content']['parts'] if not part.get('thought'))
                    usage=data.get('usageMetadata',{})
                    data={'choices':[{'message':{'content':text},'finish_reason':candidate.get('finishReason','').lower()}],
                        'model':data.get('modelVersion',self.settings.llm_model),'id':data.get('responseId',''),
                        'usage':{key:usage[value] for key,value in (('prompt_tokens','promptTokenCount'),('completion_tokens','candidatesTokenCount'),('total_tokens','totalTokenCount')) if value in usage}}
                choice = data["choices"][0]
                text = choice["message"]["content"]
                if not isinstance(text, str) or not text.strip():
                    raise ValueError
                usage = data.get("usage") or {}
                if not isinstance(usage, dict):
                    raise ValueError
                safe_usage = {k: v for k, v in usage.items()
                              if k in {"prompt_tokens", "completion_tokens", "total_tokens"}
                              and type(v) is int and v >= 0}
                details = usage.get("completion_tokens_details") or {}
                if isinstance(details, dict) and type(details.get("reasoning_tokens")) is int:
                    safe_usage["reasoning_tokens"] = details["reasoning_tokens"]
                return GenerationTrace(
                    raw_output=text, model=str(data.get("model") or self.settings.llm_model),
                    response_id=str(data.get("id") or ""),
                    finish_reason=str(choice.get("finish_reason") or ""),
                    usage=safe_usage, latency_ms=(time.perf_counter()-started)*1000, attempts=attempt,
                )
            except (ValueError, KeyError, IndexError, TypeError, AttributeError):
                raise LLMError("LLM response did not match the text Chat Completions contract", code="invalid_api_response") from None
        raise AssertionError("unreachable")
