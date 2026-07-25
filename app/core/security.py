import hmac
import hashlib

from app.core.config import GITHUB_WEBHOOK_SECRET


def verify_github_signature(raw_body: bytes, signature_header: str | None) -> bool:
    if not GITHUB_WEBHOOK_SECRET:
        raise RuntimeError("GITHUB_WEBHOOK_SECRET is not configured")

    if not signature_header:
        print("No X-Hub-Signature-256 header present", flush=True)
        return False

    expected_signature = "sha256=" + hmac.new(
        GITHUB_WEBHOOK_SECRET.encode("utf-8"),
        raw_body,
        hashlib.sha256,
    ).hexdigest()

    print("Received:", signature_header, flush=True)
    print("Expected:", expected_signature, flush=True)

    return hmac.compare_digest(expected_signature, signature_header)