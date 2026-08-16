from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.models.evidence import Evidence, RawLog
from app.services.replay_toolbelt import ReplayToolbelt


class ReplayToolbeltTests(unittest.TestCase):
    def setUp(self):
        self.evidence = Evidence(
            run_id=101,
            workflow_name="CI",
            branch="main",
            sha="abc123",
            failed_jobs=["test"],
            logs=[RawLog(job_id=1, job_name="test", text="AssertionError", lines=["AssertionError"])],
            diff="diff --git a/tests/test_example.py",
            last_green_run_id=99,
            pull_request_number=None,
        )
        self.tools = ReplayToolbelt(
            self.evidence,
            playbook_chunks=[
                {"text": "Python assertion failures indicate test mismatches", "quality_score": 1.0},
                {"text": "Missing dependencies cause import errors", "quality_score": 1.0},
            ],
        )

    def test_reads_only_current_fixture(self):
        self.assertEqual(self.tools.get_failed_logs(101), self.evidence.logs)
        self.assertEqual(self.tools.get_commit_diff("abc123"), self.evidence.diff)
        self.assertEqual(self.tools.get_last_green_run("CI", "main", 101).run_id, 99)

        with self.assertRaises(ValueError):
            self.tools.get_failed_logs(102)
        with self.assertRaises(ValueError):
            self.tools.get_commit_diff("other-sha")

    def test_rerun_is_fixture_only_and_playbook_is_deterministic(self):
        self.assertFalse(self.tools.rerun_job(101))
        chunks = self.tools.search_playbook("python assertion test", limit=1)
        self.assertEqual(len(chunks), 1)
        self.assertIn("assertion", chunks[0]["text"].lower())


if __name__ == "__main__":
    unittest.main()