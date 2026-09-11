# CI/CD Triage Agent

An evidence-grounded service that investigates failed GitHub Actions runs, produces a structured root-cause analysis (RCA), and delivers the result to GitHub and Slack.

The service is designed to fail safely: it redacts sensitive data before reasoning, verifies generated claims against collected evidence, applies a configurable confidence threshold, and abstains when it cannot support a reliable conclusion. Suspected flaky failures are rerun before they are reported as flaky.

## Features

- Receives signed GitHub `workflow_run` webhooks and queues failed runs for background processing
- Collects failed-job logs, the commit diff, pull request metadata, and the last successful run
- Redacts common secrets and credentials before sending evidence to the model
- Uses an OpenAI model and a controlled tool-first reasoning loop to generate RCAs
- Classifies failures and verifies that claims are grounded in logs or diffs
- Applies confidence gating, with explicit abstention for uncertain results
- Reruns failed jobs before confirming suspected flaky failures
- Publishes reports as pull request comments or commit comments
- Sends report summaries to Slack when Slack is configured
- Stores incidents, reasoning traces, delivery status, and reviewer feedback
- Reuses previous incidents as RAG context; optionally uses PostgreSQL with pgvector
- Exposes OpenTelemetry traces and includes Jaeger in the Docker Compose stack
- Includes unit/integration scripts and an offline replay harness for evaluation

## How It Works

```text
GitHub Actions failure
        |
        v
Signed workflow_run webhook
        |
        v
In-memory job queue and background worker
        |
        v
Evidence collection -> redaction -> historical incident retrieval
        |
        v
OpenAI RCA -> failure normalization -> grounding verification
        |
        v
Confidence gate ---- low/unknown ----> abstain safely
        |
        +---- suspected flaky --------> rerun failed jobs
        |
        v
GitHub comment + Slack message + incident storage + feedback links
```

## Tech Stack

- Python and FastAPI
- OpenAI API
- GitHub REST API and GitHub webhooks
- SQLAlchemy with PostgreSQL (SQLite can also be used locally)
- pgvector for optional vector retrieval
- OpenTelemetry and Jaeger
- Docker and Docker Compose

## Project Structure

```text
app/
  core/        Configuration, database, webhook security, and telemetry
  models/      Evidence, RCA, incident, feedback, and RAG data models
  routes/      GitHub webhook and feedback HTTP endpoints
  services/    Collection, reasoning, verification, delivery, storage, and worker logic
scripts/       Tests, replay tooling, sample exports, and manual validation scripts
Dockerfile
docker-compose.yml
requirements.txt
```

## Quick Start with Docker Compose

### 1. Configure the environment

Create a `.env` file in the project root:

```dotenv
GITHUB_TOKEN=github_pat_your_token
GITHUB_OWNER=your-github-owner
GITHUB_REPO=your-repository
GITHUB_WEBHOOK_SECRET=replace-with-a-long-random-secret

OPENAI_API_KEY=sk-your-key
OPENAI_MODEL=gpt-5

SLACK_BOT_TOKEN=xoxb-your-token
SLACK_CHANNEL_ID=C0123456789

APP_BASE_URL=https://your-public-service.example.com
RCA_CONFIDENCE_THRESHOLD=0.8
ENABLE_PGVECTOR=true
```

`SLACK_BOT_TOKEN` and `SLACK_CHANNEL_ID` are optional. Docker Compose supplies `DATABASE_URL` and the Jaeger OTLP endpoint automatically.

### 2. Start the stack

```bash
docker compose up --build
```

The following services become available:

- API: `http://localhost:8000`
- Interactive API docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`
- Jaeger UI: `http://localhost:16686`
- PostgreSQL: `localhost:5432`

Database tables are created automatically when the application starts.

## Local Development

Python 3.11 or newer is recommended. A running PostgreSQL instance is not required for basic local development; SQLite is sufficient when pgvector is disabled.

```bash
python -m venv .venv
```

Activate the virtual environment, then install dependencies:

```bash
python -m pip install -r requirements.txt
```

Create `.env` using the example above, with a local database configuration such as:

```dotenv
DATABASE_URL=sqlite:///./triage.db
ENABLE_PGVECTOR=false
```

Start the API:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## GitHub Configuration

Create a repository webhook with these settings:

- Payload URL: `https://your-public-service.example.com/webhooks/github`
- Content type: `application/json`
- Secret: the same value as `GITHUB_WEBHOOK_SECRET`
- Event: **Workflow runs**

The endpoint validates `X-Hub-Signature-256`, ignores unrelated events and successful runs, and queues only failed workflow runs.

The GitHub token must be able to read Actions runs, jobs, logs, commits, and repository metadata. It must also be able to rerun failed jobs and create pull request issue comments or commit comments. The exact fine-grained permissions depend on the token type and repository policy.

For local webhook testing, expose port 8000 through a secure tunnel and set both the GitHub webhook URL and `APP_BASE_URL` to its public HTTPS address.

## Configuration Reference

| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `GITHUB_TOKEN` | Yes | — | Authenticates GitHub API calls |
| `GITHUB_OWNER` | Yes | — | Owner of the monitored repository |
| `GITHUB_REPO` | Yes | — | Name of the monitored repository |
| `GITHUB_WEBHOOK_SECRET` | Yes | — | Verifies webhook HMAC signatures |
| `OPENAI_API_KEY` | Yes | — | Authenticates model requests |
| `OPENAI_MODEL` | No | `gpt-5` | Model used for RCA generation |
| `DATABASE_URL` | Yes | — | SQLAlchemy database URL |
| `SLACK_BOT_TOKEN` | No | — | Slack bot token for notifications |
| `SLACK_CHANNEL_ID` | No | — | Destination Slack channel |
| `APP_BASE_URL` | No | `http://localhost:8000` | Base URL used in feedback links |
| `RCA_CONFIDENCE_THRESHOLD` | No | `0.8` | Minimum confidence for delivery |
| `ENABLE_PGVECTOR` | No | `false` | Enables 32-dimensional pgvector storage/search |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | No | `http://localhost:4318/v1/traces` | OTLP/HTTP trace collector endpoint |

Never commit `.env` or real credentials to source control.

## API Endpoints

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/` | Service identity and status |
| `GET` | `/health` | Health check |
| `POST` | `/webhooks/github` | Signed GitHub webhook receiver |
| `POST` | `/feedback` | Records structured reviewer feedback |
| `GET` | `/feedback/capture/{incident_id}/{verdict}` | One-click feedback from a report |
| `GET` | `/feedback/{feedback_id}` | Returns one feedback record |
| `GET` | `/feedback/incident/{incident_id}` | Lists feedback for an incident |

Supported feedback verdicts are `correct`, `wrong`, and `not_quite`. Feedback adjusts the quality score of the associated incident in the retrieval store.

## Safety and Decision Policy

The RCA pipeline normalizes model output, then checks every generated claim against the collected evidence. If the first response is not grounded, it retries once with stricter grounding instructions. A second failure produces an explicit abstention.

Even a grounded RCA is withheld when its class is unknown, its confidence is invalid, or its confidence falls below `RCA_CONFIDENCE_THRESHOLD`. A `flaky` classification triggers a failed-job rerun; the agent confirms flakiness only if the new attempt succeeds without a code change.

## Testing

Most automated tests are standard-library `unittest` modules under `scripts/`. Configure `DATABASE_URL` before test discovery because database-backed modules load it at import time.

```bash
python -m unittest discover -s scripts -p "test_*_unit.py"
```

You can also run individual tests:

```bash
python scripts/test_redactor.py
python scripts/test_openai_reasoner.py
python scripts/test_triage_service.py
```

Some scripts are integration or manual delivery checks and can call live GitHub, OpenAI, Slack, or database services. Review a script before running it with production credentials—especially files whose names contain `real`, `delivery`, `pipeline`, or `service`.

## Replay Evaluation

The replay harness evaluates saved evidence snapshots without fetching live GitHub data. It reports classification accuracy, abstention accuracy, grounding pass rate, per-class accuracy, and the decision breakdown.

```bash
python scripts/replay_harness.py \
  --sample-file scripts/replay_samples.json \
  --output-file scripts/replay_report.json
```

Replay reasoning still uses the configured OpenAI model, while GitHub-facing tools are replaced with replay implementations.

## Operational Notes

- The job queue is in memory. Pending jobs are lost when the process restarts, and the current design is intended for a single application process.
- Schema creation and small compatibility migrations run at startup; there is no external migration framework yet.
- When pgvector is unavailable or disabled, retrieval falls back to SQL keyword matching.
- Traces are written to the console and exported through OTLP/HTTP.
- RCA results and delivery failures are persisted as incidents for audit and future retrieval.

## License

No license file is currently included. Add one before distributing or accepting external contributions.
