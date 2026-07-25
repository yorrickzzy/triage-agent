from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.confidence_gate import ConfidenceGate

gate = ConfidenceGate(threshold=0.8)

cases = [
    {
        "failure_class": "test",
        "confidence": 0.96,
        "abstained": False,
    },
    {
        "failure_class": "test",
        "confidence": 0.55,
        "abstained": False,
    },
    {
        "failure_class": "unknown",
        "confidence": 0.95,
        "abstained": False,
    },
    {
        "failure_class": "flaky",
        "confidence": 0.90,
        "abstained": False,
    },
    {
        "failure_class": "test",
        "confidence": 0.99,
        "abstained": True,
    },
]

for case in cases:
    result = gate.evaluate(case)

    print(
        f"class={case['failure_class']}, "
        f"confidence={case['confidence']}, "
        f"abstained={case['abstained']} "
        f"-> {result.decision}: {result.reason}"
    )