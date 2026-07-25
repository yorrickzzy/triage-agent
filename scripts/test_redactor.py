from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.redactor import Redactor

redactor = Redactor()

sample = """
user email: zeyuan@example.com
GITHUB_TOKEN=ghp_abcd1234567890
TOKEN=github_pat_11ABCDEF_some_secret_value
Authorization: Bearer abc.def.ghi
api_key=sk-test-123456
password=my-password
secret: very-secret-value
"""

print(redactor.redact_text(sample))