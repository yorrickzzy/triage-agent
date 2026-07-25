import re


REDACTION_PATTERNS = [
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "<EMAIL>"),
    (re.compile(r"ghp_[A-Za-z0-9_]+"), "<GITHUB_TOKEN>"),
    (re.compile(r"github_pat_[A-Za-z0-9_]+"), "<GITHUB_TOKEN>"),
    (re.compile(r"Bearer\s+[A-Za-z0-9._\-]+"), "Bearer <TOKEN>"),
    (re.compile(r"(?i)(api[_-]?key\s*[:=]\s*)[^\s]+"), r"\1<API_KEY>"),
    (re.compile(r"(?i)(password\s*[:=]\s*)[^\s]+"), r"\1<PASSWORD>"),
    (re.compile(r"(?i)(secret\s*[:=]\s*)[^\s]+"), r"\1<SECRET>"),
]


class Redactor:
    def redact_text(self, text: str) -> str:
        redacted = text

        for pattern, replacement in REDACTION_PATTERNS:
            redacted = pattern.sub(replacement, redacted)

        return redacted