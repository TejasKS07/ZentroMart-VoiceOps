import unittest
from unittest.mock import Mock
from src.rate_limit import WeightedRateLimiter,retry_delay


class FakeClock:
    def __init__(self):self.now=0;self.sleeps=[]
    def time(self):return self.now
    def sleep(self,seconds):self.sleeps.append(seconds);self.now+=seconds


class RateLimitTests(unittest.TestCase):
    def test_embeddings_share_weighted_budget_with_answers_and_grading(self):
        clock=FakeClock();limit=WeightedRateLimiter(clock=clock.time,sleep=clock.sleep)
        limit.acquire(52)
        for _ in range(38):limit.acquire()
        self.assertEqual(clock.now,0)
        limit.acquire()
        self.assertGreaterEqual(clock.now,60)
        self.assertLessEqual(limit.peak,90)
        self.assertTrue(all(s<=1 for s in clock.sleeps))

    def test_cooldown_honors_long_server_retry_delay_without_a_burst(self):
        clock=FakeClock();limit=WeightedRateLimiter(clock=clock.time,sleep=clock.sleep)
        limit.cooldown(125);limit.acquire()
        self.assertGreaterEqual(clock.now,125)
        response=Mock(headers={'Retry-After':'72'})
        response.json.return_value={'error':{'details':[{'retryDelay':'81.5s'}]}}
        self.assertEqual(retry_delay(response),81.5)

    def test_oversized_batch_is_rejected_instead_of_overrunning_rpm(self):
        limit=WeightedRateLimiter()
        with self.assertRaises(ValueError):limit.acquire(100)

    def test_approved_attempt_cap_counts_reservations_before_network_calls(self):
        limit=WeightedRateLimiter(max_attempts=2)
        limit.acquire();limit.acquire()
        with self.assertRaisesRegex(RuntimeError,'budget exhausted'):limit.acquire()
        self.assertEqual(limit.summary()['reserved_attempts'],2)
