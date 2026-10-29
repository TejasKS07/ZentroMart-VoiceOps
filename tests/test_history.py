from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.history import HistoryStore


class HistoryTests(unittest.TestCase):
    def test_saved_history_survives_restart_and_retries_replace_one_turn(self):
        with TemporaryDirectory() as folder:
            path=Path(folder)/'history.sqlite3'
            first=HistoryStore(path)
            first.create('conversation','owner')
            first.append('conversation',{'display_message':'[REDACTED]','response':{'reply':'Verified','diagnostics':{'intent':'order'}}},'request')
            first.append('conversation',{'display_message':'[REDACTED]','response':{'reply':'Verified again','diagnostics':{'intent':'order'}}},'request')
            restarted=HistoryStore(path)
            self.assertEqual(restarted.get('conversation','owner')['messages'][0]['response']['reply'],'Verified again')
            self.assertEqual(len(restarted.get('conversation','owner')['messages']),1)
            self.assertIsNone(restarted.get('conversation','other'))
            self.assertEqual(restarted.list('other'),[])
            first.connection.close();restarted.connection.close()
