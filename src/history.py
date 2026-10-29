"""Owner-scoped, sanitized chat history; never stores verification state."""

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from threading import RLock


class HistoryStore:
    def __init__(self, path=None):
        if path:
            path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        self.connection=sqlite3.connect(str(path) if path else ':memory:',check_same_thread=False)
        self.lock=RLock()
        with self.connection:
            self.connection.execute('CREATE TABLE IF NOT EXISTS conversations (id TEXT PRIMARY KEY, owner TEXT NOT NULL, title TEXT NOT NULL, created TEXT NOT NULL, updated TEXT NOT NULL, messages TEXT NOT NULL)')

    def create(self, conversation, owner):
        now=datetime.now(timezone.utc).isoformat()
        with self.lock,self.connection:
            self.connection.execute('INSERT INTO conversations VALUES (?,?,?,?,?,?)',(conversation,owner,'New conversation',now,now,'[]'))

    def append(self, conversation, result, request_id=None):
        # Only call this with the gateway-sanitized browser response.
        with self.lock,self.connection:
            row=self.connection.execute('SELECT messages,title FROM conversations WHERE id=?',(conversation,)).fetchone()
            if not row:return
            messages=json.loads(row[0])
            existing=next((i for i,m in enumerate(messages) if request_id and m.get('request_id')==request_id),None)
            created=messages[existing].get('created_at') if existing is not None else datetime.now(timezone.utc).isoformat()
            entry={**result,'request_id':request_id,'created_at':created}
            if existing is None:messages.append(entry)
            else:messages[existing]=entry
            diagnostic=result.get('response',{}).get('diagnostics',{})
            titles={'order':'Order support','tracking':'Order tracking','refund':'Refund question','return':'Return request','coupon':'Coupon question','human':'Human support','new_order':'New order','cancel':'Cancellation','serviceability':'Delivery availability','damage':'Damaged item','safety':'Product safety','repeat_contact':'Support follow-up'}
            title=row[1] if row[1]!='New conversation' else titles.get(diagnostic.get('intent'),'Support conversation')
            self.connection.execute('UPDATE conversations SET title=?,updated=?,messages=? WHERE id=?',(title,datetime.now(timezone.utc).isoformat(),json.dumps(messages[-100:],ensure_ascii=False),conversation))

    def list(self, owner):
        with self.lock:
            return [dict(zip(('id','title','created','updated'),r)) for r in self.connection.execute("SELECT id,title,created,updated FROM conversations WHERE owner=? AND messages!='[]' ORDER BY updated DESC LIMIT 50",(owner,))]

    def get(self, conversation, owner):
        with self.lock:
            row=self.connection.execute('SELECT title,messages FROM conversations WHERE id=? AND owner=?',(conversation,owner)).fetchone()
            return {'id':conversation,'title':row[0],'messages':json.loads(row[1])} if row else None

    def delete(self, conversation, owner):
        with self.lock,self.connection:
            return self.connection.execute('DELETE FROM conversations WHERE id=? AND owner=?',(conversation,owner)).rowcount==1
