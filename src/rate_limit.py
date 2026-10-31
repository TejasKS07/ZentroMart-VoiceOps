"""Shared weighted sliding-window pacing for the authorized evaluation run."""

from collections import deque
from datetime import datetime,timezone
from email.utils import parsedate_to_datetime
import json
from threading import RLock
import time


def retry_delay(response, fallback=60):
    """Honor Retry-After and Google's RetryInfo without exposing response text."""
    delays=[float(fallback)]
    header=response.headers.get('Retry-After') if hasattr(response,'headers') else None
    if isinstance(header,str):
        try:delays.append(float(header))
        except ValueError:
            try:delays.append(parsedate_to_datetime(header).timestamp()-time.time())
            except (TypeError,ValueError,OverflowError):pass
    try:
        for detail in response.json().get('error',{}).get('details',[]):
            value=detail.get('retryDelay')
            if value:delays.append(float(str(value).rstrip('s')))
    except (ValueError,TypeError,AttributeError):pass
    return max(0,*delays)


class WeightedRateLimiter:
    def __init__(self, *, capacity=90, window=60, log_path=None, clock=time.monotonic, sleep=time.sleep,max_attempts=None):
        self.capacity,self.window=capacity,window
        self.clock,self.sleep=clock,sleep
        self.events=deque();self.lock=RLock();self.blocked_until=0
        self.peak=0;self.records=[];self.log_path=log_path
        self.max_attempts=max_attempts;self.issued=0

    def acquire(self,weight=1):
        if not 1<=weight<=self.capacity:raise ValueError('Request weight exceeds the RPM safety budget')
        while True:
            with self.lock:
                if self.max_attempts is not None and self.issued>=self.max_attempts:
                    raise RuntimeError('Approved provider-attempt budget exhausted')
                now=self.clock()
                while self.events and self.events[0][0]<=now-self.window:self.events.popleft()
                total=sum(item[1] for item in self.events)
                delay=max(0,self.blocked_until-now)
                if total+weight>self.capacity:
                    needed=total+weight-self.capacity;released=0
                    for started,units in self.events:
                        released+=units
                        if released>=needed:
                            delay=max(delay,started+self.window-now+0.01);break
                if delay<=0:
                    self.events.append((now,weight));self.peak=max(self.peak,total+weight)
                    self.issued+=1
                    return datetime.now(timezone.utc).isoformat()
            self.sleep(min(delay,1))

    def cooldown(self,seconds):
        with self.lock:self.blocked_until=max(self.blocked_until,self.clock()+seconds)

    def record(self, *, started,kind,weight,status):
        row={'started_at':started,'kind':kind,'quota_weight':weight,'status':status}
        with self.lock:
            self.records.append(row)
            if self.log_path:
                with self.log_path.open('a',encoding='utf-8') as handle:handle.write(json.dumps(row)+'\n')

    def summary(self):
        with self.lock:
            return {'user_rpm_limit':100,'safety_capacity':self.capacity,'window_seconds':self.window,
                'approved_attempt_limit':self.max_attempts,'reserved_attempts':self.issued,
                'peak_quota_weight_in_window':self.peak,'http_attempts':len(self.records),
                'successful_http_requests':sum(r['status']==200 for r in self.records),
                'embedding_http_requests':sum(r['kind']=='embedding' for r in self.records),
                'chat_completion_http_requests':sum(r['kind']=='chat_completion' for r in self.records),
                'rate_limit_responses':sum(r['status']==429 for r in self.records)}
