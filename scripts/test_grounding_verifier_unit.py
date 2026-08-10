from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.models.evidence import Evidence, RawLog
from app.services.grounding_verifier import GroundingVerifier


class GroundingVerifierUnitTests(unittest.TestCase):
    def test_verify_accepts_quote_with_line_prefix_timestamp_and_control_noise(self):
        raw_log = RawLog(
            job_id=1,
            job_name="test",
            text="",
            lines=[
                "2026-08-09T10:58:14.6812199Z   File \"/home/runner/work/x/calculator.py\", line 6",
                "2026-08-09T10:58:14.6813245Z     print(\"hello\"",
                "2026-08-09T10:58:14.6814144Z SyntaxError: '(' was never closed",
            ],
        )
        evidence = Evidence(
            run_id=1,
            workflow_name="CI",
            branch="main",
            sha="abc",
            failed_jobs=["test"],
            logs=[raw_log],
            diff="",
            last_green_run_id=None,
            pull_request_number=None,
        )

        noisy_quote = (
            "167 | 2026-08-09T10:58:14.6812199Z   File \"/home/runner/work/x/calculator.py\", line 6\\u0094"
            "\n168 | 2026-08-09T10:58:14.6813245Z     print(\"hello\"\\u0094"
            "\n169 | 2026-08-09T10:58:14.6814144Z SyntaxError: '(' was never closed"
        )

        rca = {
            "claims": [
                {
                    "statement": "syntax error in calculator.py",
                    "citations": [
                        {
                            "line_start": 1,
                            "line_end": 3,
                            "quote": noisy_quote,
                        }
                    ],
                }
            ]
        }

        verifier = GroundingVerifier()
        self.assertTrue(verifier.verify(rca, evidence))


if __name__ == "__main__":
    unittest.main()
