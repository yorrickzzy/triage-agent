from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.failure_classifier import FailureClassifier


def main() -> list[dict]:
    classifier = FailureClassifier()

    cases = [
        {"failure_class": "test"},
        {"failure_class": "TEST"},
        {"failure_class": "Testing"},
        {"failure_class": "pytest"},
        {"failure_class": "lint"},
        {"failure_class": "docker"},
    ]

    results = []
    for case in cases:
        result = classifier.normalize(case)
        print(result)
        results.append(result)

    return results


if __name__ == "__main__":
    results = main()
    print("returned:", results)