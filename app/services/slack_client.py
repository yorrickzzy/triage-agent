import os
import requests


class SlackClient:

    def __init__(self):
        self.token = os.getenv("SLACK_BOT_TOKEN")
        self.channel_id = os.getenv("SLACK_CHANNEL_ID")

        if not self.token:
            raise ValueError("Missing SLACK_BOT_TOKEN")

        if not self.channel_id:
            raise ValueError("Missing SLACK_CHANNEL_ID")


    def send_message(self, message: str):

        url = "https://slack.com/api/chat.postMessage"

        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }

        payload = {
            "channel": self.channel_id,
            "text": message
        }

        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=20,
        )

        response.raise_for_status()

        data = response.json()

        if not data.get("ok"):
            raise Exception(
                f"Slack API error: {data}"
            )

        return data
