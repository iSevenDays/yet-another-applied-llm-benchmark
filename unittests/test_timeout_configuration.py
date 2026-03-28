#!/usr/bin/env python3
import unittest

import llm
import main


class TestTimeoutConfiguration(unittest.TestCase):
    def test_worker_timeout_is_one_hour(self):
        self.assertEqual(main.MAX_WAIT_TIME_SECONDS, 3600)

    def test_stream_timeouts_are_one_hour(self):
        self.assertEqual(llm.STREAM_CHUNK_TIMEOUT_SECONDS, 3600)
        self.assertEqual(llm.REQUEST_OVERALL_TIMEOUT_SECONDS, 3600)
        self.assertEqual(llm.OPENAI_CLIENT_TIMEOUT_SECONDS, 3600)


if __name__ == "__main__":
    unittest.main()
