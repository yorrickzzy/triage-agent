from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from app.models.evidence import Evidence, RawLog
from app.services.confidence_gate import ConfidenceGate
from app.services.failure_classifier import FailureClassifier
from app.services.grounding_verifier import GroundingVerifier
from app.services.openai_reasoner import OpenAIReasoner
from app.services.rca_pipeline import RCAPipeline
from app.services.replay_toolbelt import ReplayToolbelt


@dataclass
class ReplaySample:
    run_id: int
    evidence: dict[str, Any]
    expected_failure_class: str | None = None
    expected_abstain: bool | None = None
    true_cause: str | None = None
    ambiguous: bool | None = None
    note: str | None = None
    playbook_chunks: list[dict[str, Any]] | None = None
    rerun_accepted: bool = False


@dataclass
class ReplayResult:
    run_id: int
    expected_failure_class: str | None
    predicted_failure_class: str
    expected_abstain: bool | None
    model_abstained: bool
    final_abstained: bool
    grounding_passed: bool
    confidence: float
    gate_decision: str
    classification_match: bool | None
    abstain_match: bool | None
    true_cause: str | None = None
    ambiguous: bool | None = None
    error: str | None = None
    note: str | None = None


def _load_samples(path: Path) -> list[ReplaySample]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    raw_samples = data.get("samples", [])

    samples: list[ReplaySample] = []
    for item in raw_samples:
        samples.append(
            ReplaySample(
                run_id=int(item["run_id"]),
                evidence=dict(item["evidence"]),
                expected_failure_class=item.get("expected_failure_class"),
                expected_abstain=item.get("expected_abstain"),
                true_cause=item.get("true_cause"),
                ambiguous=item.get("ambiguous"),
                note=item.get("note"),
                playbook_chunks=item.get("playbook_chunks"),
                rerun_accepted=bool(item.get("rerun_accepted", False)),
            )
        )

    return samples


def _raw_log_from_dict(item: dict[str, Any]) -> RawLog:
    text = str(item["text"])
    lines = item.get("lines")
    if not isinstance(lines, list):
        lines = text.splitlines()

    return RawLog(
        job_id=int(item["job_id"]),
        job_name=str(item["job_name"]),
        text=text,
        lines=[str(line) for line in lines],
    )


def _evidence_from_dict(item: dict[str, Any]) -> Evidence:
    logs = [_raw_log_from_dict(log_item) for log_item in item.get("logs", [])]

    return Evidence(
        run_id=int(item["run_id"]),
        workflow_name=str(item["workflow_name"]),
        branch=str(item["branch"]),
        sha=str(item["sha"]),
        failed_jobs=[str(job) for job in item.get("failed_jobs", [])],
        logs=logs,
        diff=str(item.get("diff", "")),
        last_green_run_id=item.get("last_green_run_id"),
        pull_request_number=item.get("pull_request_number"),
    )


def _build_pipeline(
    evidence: Evidence,
    sample: ReplaySample,
) -> tuple[RCAPipeline, ConfidenceGate]:
    tools = ReplayToolbelt(
        evidence=evidence,
        playbook_chunks=sample.playbook_chunks,
        rerun_accepted=sample.rerun_accepted,
    )
    reasoner = OpenAIReasoner(tools=tools)
    classifier = FailureClassifier()
    verifier = GroundingVerifier()

    pipeline = RCAPipeline(
        reasoner=reasoner,
        classifier=classifier,
        verifier=verifier,
    )
    confidence_gate = ConfidenceGate()

    return pipeline, confidence_gate


def _run_sample(sample: ReplaySample) -> ReplayResult:
    evidence = _evidence_from_dict(sample.evidence)
    try:
        pipeline, confidence_gate = _build_pipeline(evidence, sample)
        rca = pipeline.run(evidence)
        gate_result = confidence_gate.evaluate(rca)
        grounding_passed = pipeline.verifier.verify(rca, evidence)

        predicted_failure_class = str(rca.get("failure_class", "unknown"))
        model_abstained = bool(rca.get("abstained", False))
        confidence = float(rca.get("confidence", 0.0))
        final_abstained = gate_result.decision == "abstain"

        classification_match = None
        if sample.expected_failure_class is not None:
            classification_match = (
                sample.expected_failure_class == predicted_failure_class
            )

        abstain_match = None
        if sample.expected_abstain is not None:
            abstain_match = sample.expected_abstain == final_abstained

        return ReplayResult(
            run_id=sample.run_id,
            expected_failure_class=sample.expected_failure_class,
            predicted_failure_class=predicted_failure_class,
            expected_abstain=sample.expected_abstain,
            model_abstained=model_abstained,
            final_abstained=final_abstained,
            grounding_passed=grounding_passed,
            confidence=confidence,
            gate_decision=gate_result.decision,
            classification_match=classification_match,
            abstain_match=abstain_match,
            true_cause=sample.true_cause,
            ambiguous=sample.ambiguous,
            note=sample.note,
        )
    except Exception as exc:
        return ReplayResult(
            run_id=sample.run_id,
            expected_failure_class=sample.expected_failure_class,
            predicted_failure_class="unknown",
            expected_abstain=sample.expected_abstain,
            model_abstained=True,
            final_abstained=True,
            grounding_passed=False,
            confidence=0.0,
            gate_decision="abstain",
            classification_match=False
            if sample.expected_failure_class is not None
            else None,
            abstain_match=(
                sample.expected_abstain is True
                if sample.expected_abstain is not None
                else None
            ),
            true_cause=sample.true_cause,
            ambiguous=sample.ambiguous,
            error=str(exc),
            note=sample.note,
        )


def _print_result(result: ReplayResult) -> None:
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))


def _summarize(results: list[ReplayResult]) -> dict[str, Any]:
    total = len(results)
    if total == 0:
        return {
            "total": 0,
            "classification_accuracy": 0.0,
            "abstain_accuracy": 0.0,
            "grounding_pass_rate": 0.0,
            "grounding_fabrication_count": 0,
            "grounding_zero_fabrication": True,
            "per_class_accuracy": {},
            "decision_breakdown": {},
            "ambiguous_abstain_accuracy": 0.0,
        }

    classified = [r for r in results if r.expected_failure_class is not None]
    abstain_labeled = [r for r in results if r.expected_abstain is not None]
    ambiguous_labeled = [r for r in results if r.ambiguous is True]

    classification_accuracy = 0.0
    if classified:
        classification_accuracy = sum(
            1 for r in classified if r.classification_match is True
        ) / len(classified)

    abstain_accuracy = 0.0
    if abstain_labeled:
        abstain_accuracy = sum(
            1 for r in abstain_labeled if r.abstain_match is True
        ) / len(abstain_labeled)

    ambiguous_abstain_accuracy = 0.0
    if ambiguous_labeled:
        ambiguous_abstain_accuracy = sum(
            1 for r in ambiguous_labeled if r.final_abstained
        ) / len(ambiguous_labeled)

    grounding_pass_count = sum(1 for r in results if r.grounding_passed)
    grounding_fail_count = total - grounding_pass_count
    grounding_pass_rate = grounding_pass_count / total

    per_class_counts: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "total": 0,
            "correct": 0,
            "accuracy": 0.0,
        }
    )
    for result in classified:
        cls = str(result.expected_failure_class)
        per_class_counts[cls]["total"] += 1
        if result.classification_match:
            per_class_counts[cls]["correct"] += 1

    for cls in per_class_counts:
        total_cls = per_class_counts[cls]["total"]
        correct_cls = per_class_counts[cls]["correct"]
        per_class_counts[cls]["accuracy"] = (
            correct_cls / total_cls if total_cls else 0.0
        )

    decision_breakdown = dict(Counter(r.gate_decision for r in results))

    return {
        "total": total,
        "classification_accuracy": classification_accuracy,
        "abstain_accuracy": abstain_accuracy,
        "grounding_pass_rate": grounding_pass_rate,
        "grounding_fabrication_count": grounding_fail_count,
        "grounding_zero_fabrication": grounding_fail_count == 0,
        "per_class_accuracy": dict(per_class_counts),
        "decision_breakdown": decision_breakdown,
        "ambiguous_abstain_accuracy": ambiguous_abstain_accuracy,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local-only replay harness for CI/CD triage results."
    )
    parser.add_argument(
        "--sample-file",
        type=Path,
        required=True,
        help="JSON file with a samples array that includes full evidence snapshots.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        help="Optional JSON output file for full replay results and summary.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    samples = _load_samples(args.sample_file)

    if not samples:
        print(
            "No replay samples found in the sample file.",
            file=sys.stderr,
        )
        return 1

    results = []
    for sample in samples:
        result = _run_sample(sample)
        results.append(result)
        _print_result(result)

    summary = _summarize(results)
    payload = {
        "summary": summary,
        "results": [asdict(result) for result in results],
    }

    print(json.dumps({"summary": summary}, ensure_ascii=False, indent=2))

    if args.output_file:
        args.output_file.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Wrote replay report: {args.output_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())