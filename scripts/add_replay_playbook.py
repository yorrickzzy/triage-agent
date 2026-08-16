import json
from pathlib import Path

p = Path(__file__).resolve().parent / 'replay_samples.json'
data = json.loads(p.read_text(encoding='utf-8-sig'))
samples = data.get('samples', [])

templates = {
    'build': [
        'Build failure caused by Python syntax errors in calculator.py or project scripts. SyntaxError, IndentationError, or unclosed parentheses occur before tests run, so classify as build rather than test.',
        'When the job fails during Python execution or import before pytest starts, treat it as a build failure. Common clues are SyntaxError, invalid indentation, and stack traces pointing to a source file.',
    ],
    'dependency': [
        'Dependency issue when the environment cannot import a required package or module. ModuleNotFoundError, ImportError, or missing requirements indicate dependency problems rather than application logic regression.',
        'If the failure occurs during test collection or python startup and names a third-party module, classify as dependency. Check requirements.txt, virtualenv setup, and package installation.',
    ],
    'test': [
        'Test failure caused by a functional assertion mismatch. Pytest reports AssertionError, expected value differs from actual value, and the job reaches the test phase successfully.',
        'Use test classification when the application code runs and a unit or integration test fails on an assertion or expected output. This is a code regression, not infrastructure or build breakage.',
    ],
    'flaky': [
        'Flaky CI failure when the same job intermittently passes or fails without code changes. Rerun is required before making a final RCA; do not classify as deterministic build or test failure without repeated evidence.',
        'Intermittent failures, timeout variance, concurrency races, or nondeterministic assertions indicate flakiness. Investigate rerun behavior before final categorization.',
    ],
    'lint': [
        'Lint or formatting failure when a static checker rejects code style, formatting, or grammar issues before tests run. Treat as lint if the error cites formatting or style enforcement.',
        'Static analysis failures such as formatting violations or lint rules indicate lint classification rather than runtime logic regressions.',
    ],
    'format': [
        'Formatting failure caused by code style checks or formatter diffs. Use format classification when the log shows black, prettier, yapf, or an equivalent formatter rejecting the patch.',
        'A format issue is identified before runtime execution and can usually be resolved by applying the project formatter.',
    ],
}

for sample in samples:
    failure_class = sample.get('expected_failure_class')
    if failure_class in templates:
        sample['playbook_chunks'] = [
            {
                'source_type': 'playbook',
                'source_id': f'{failure_class}-{i}',
                'workflow_name': 'CI',
                'failure_class': failure_class,
                'quality_score': 0.95,
                'text': text,
            }
            for i, text in enumerate(templates[failure_class], start=1)
        ]
    else:
        sample['playbook_chunks'] = []

p.write_text(json.dumps(data, ensure_ascii=False, indent=4), encoding='utf-8')
print(f'updated {len(samples)} samples with playbook_chunks')
print(f'nonempty: {sum(1 for s in samples if s.get("playbook_chunks"))}')
