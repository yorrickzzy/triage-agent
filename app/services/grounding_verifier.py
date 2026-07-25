import re

TIMESTAMP_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}T.*?Z\s+"
)

class GroundingVerifier:

    def _normalize(self, text: str) -> str:
        lines = []

        for line in text.splitlines():
            # 去掉 GitHub Actions 前面的时间戳
            line = TIMESTAMP_PATTERN.sub("", line)
            lines.append(line.rstrip())

        return "\n".join(lines)

    def verify(self, rca: dict, raw_log) -> bool:
        for claim in rca.get("claims", []):
            for citation in claim.get("citations", []):

                window = raw_log.text_between(
                    citation["line_start"],
                    citation["line_end"],
                )

                normalized_window = self._normalize(window)
                normalized_quote = self._normalize(
                    citation["quote"]
                )

                if normalized_quote not in normalized_window:
                    return False

        return True