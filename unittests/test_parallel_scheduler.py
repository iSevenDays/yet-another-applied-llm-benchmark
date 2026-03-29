#!/usr/bin/env python3
import unittest
from types import SimpleNamespace
from unittest import mock

import main


class FakeAsyncResult:
    def __init__(self, ready=False, payload=None):
        self._ready = ready
        self._payload = payload

    def ready(self):
        return self._ready

    def get(self, timeout=None):
        return self._payload


class FakePool:
    def __init__(self):
        self.submitted = []

    def apply_async(self, fn, args):
        result = FakeAsyncResult()
        self.submitted.append((fn, args, result))
        return result


class FakePbar:
    def __init__(self):
        self.messages = []
        self.updates = []
        self.descriptions = []

    def write(self, msg):
        self.messages.append(msg)

    def update(self, n):
        self.updates.append(n)

    def set_description_str(self, desc):
        self.descriptions.append(desc)

    def refresh(self):
        pass


class TestParallelScheduler(unittest.TestCase):
    def test_stage_hang_threshold_scales_with_max_wait_time(self):
        self.assertEqual(main._stage_hang_threshold_seconds(3600), 1800)
        self.assertEqual(main._stage_hang_threshold_seconds(1200), 600)

    def test_stage_hang_threshold_never_exceeds_timeout_budget(self):
        self.assertEqual(main._stage_hang_threshold_seconds(300), 299)
        self.assertEqual(main._stage_hang_threshold_seconds(61), 60)

    def test_format_parallel_status_limits_list_and_reports_queue(self):
        active_jobs = {
            0: {"async_result": FakeAsyncResult(ready=False), "started_at": 0.0, "test_name": "t0"},
            1: {"async_result": FakeAsyncResult(ready=False), "started_at": 10.0, "test_name": "t1"},
            2: {"async_result": FakeAsyncResult(ready=False), "started_at": 20.0, "test_name": "t2"},
            3: {"async_result": FakeAsyncResult(ready=False), "started_at": 30.0, "test_name": "t3"},
            4: {"async_result": FakeAsyncResult(ready=False), "started_at": 40.0, "test_name": "t4"},
            5: {"async_result": FakeAsyncResult(ready=False), "started_at": 50.0, "test_name": "t5"},
        }

        summary = main._format_parallel_status(active_jobs, 100.0, queued_jobs=12, max_listed_jobs=3)

        self.assertIn("6 active", summary)
        self.assertIn("12 queued", summary)
        self.assertIn("Slowest:", summary)
        self.assertIn("t0(100s)", summary)
        self.assertIn("t1(90s)", summary)
        self.assertIn("t2(80s)", summary)
        self.assertIn("(+3 more)", summary)
        self.assertNotIn("t5(50s)", summary)

    def test_format_parallel_status_handles_no_running_jobs(self):
        summary = main._format_parallel_status({}, 100.0, queued_jobs=4)
        self.assertEqual(summary, "PARALLEL: no active running jobs, 4 queued")

    def test_fill_parallel_worker_slots_respects_parallelism_limit(self):
        pool = FakePool()
        active_jobs = {}
        test_data = [("t1",), ("t2",), ("t3",)]

        next_index = main._fill_parallel_worker_slots(pool, test_data, 0, active_jobs, parallel_workers=2)

        self.assertEqual(next_index, 2)
        self.assertEqual(len(active_jobs), 2)
        self.assertEqual(active_jobs[0]["test_name"], "t1")
        self.assertEqual(active_jobs[1]["test_name"], "t2")

    def test_fill_parallel_worker_slots_refills_only_open_slot(self):
        pool = FakePool()
        active_jobs = {
            0: {"async_result": FakeAsyncResult(), "started_at": 10.0, "test_name": "t1"}
        }
        test_data = [("t1",), ("t2",), ("t3",)]

        next_index = main._fill_parallel_worker_slots(pool, test_data, 1, active_jobs, parallel_workers=2)

        self.assertEqual(next_index, 2)
        self.assertEqual(sorted(job["test_name"] for job in active_jobs.values()), ["t1", "t2"])

    def test_check_completed_jobs_processes_ready_results_and_removes_job(self):
        pbar = FakePbar()
        total_passed = [0]
        total_failed = [0]
        sr = {}
        active_jobs = {
            1: {
                "async_result": FakeAsyncResult(ready=True, payload=("demo", True, "ok")),
                "started_at": 100.0,
                "test_name": "demo",
            }
        }

        with mock.patch.object(main, "_log_test_outcome") as log_outcome:
            found = main._check_completed_jobs(active_jobs, 105.0, total_passed, total_failed, sr, pbar)

        self.assertTrue(found)
        self.assertEqual(active_jobs, {})
        self.assertEqual(total_passed, [1])
        self.assertEqual(total_failed, [0])
        self.assertEqual(sr["demo"], (True, "ok"))
        log_outcome.assert_called_once()

    def test_check_completed_jobs_marks_safety_timeout_from_active_runtime(self):
        pbar = FakePbar()
        total_passed = [0]
        total_failed = [0]
        sr = {}
        active_jobs = {
            2: {
                "async_result": FakeAsyncResult(ready=False),
                "started_at": 0.0,
                "test_name": "slow_test",
            }
        }

        found = main._check_completed_jobs(
            active_jobs,
            main.MAX_WAIT_TIME_SECONDS + 61,
            total_passed,
            total_failed,
            sr,
            pbar,
        )

        self.assertTrue(found)
        self.assertEqual(active_jobs, {})
        self.assertEqual(total_failed, [1])
        self.assertIn("slow_test", sr)
        self.assertTrue(any("TIMEOUT: slow_test" in message for message in pbar.messages))


if __name__ == "__main__":
    unittest.main()
