from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv()

from app.services.slack_client import SlackClient


client = SlackClient()


result = client.send_message(
    """
🔍 Chamber Test Message

Slack integration works!

Status:
OK
"""
)

print(result)