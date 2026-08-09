import re

TIMESTAMP_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}T.*?Z\s+"
)
# Strips "123 | " line-number prefix the model copies from numbered log output.
LINE_NUMBER_PREFIX = re.compile(r"^\d+\s*\|\s*")

class GroundingVerifier:

    def _normalize(self, text: str) -> str:
        lines = []

        for line in text.splitlines():
            line = LINE_NUMBER_PREFIX.sub("", line)
            line = TIMESTAMP_PATTERN.sub("", line)
            lines.append(line.rstrip())

        return "\n".join(lines)

    def verify(self, rca: dict, evidence) -> bool:
        """
        Verify that every citation in the RCA points to real log text.
        
        Args:
            rca: The RCA dict with claims containing citations
            evidence: Evidence object containing logs (List[RawLog])
        
        Returns:
            True if all citations are valid and grounded, False otherwise
        """
        for claim in rca.get("claims", []):
            for citation in claim.get("citations", []):
                # First validate citation structure
                if not self._validate_citation(citation):
                    print(
                        f"[grounding] citation validation failed: {citation}",
                        flush=True,
                    )
                    return False

                # Try to find the citation in any log
                found = False
                for raw_log in evidence.logs:
                    if self._verify_in_log(citation, raw_log):
                        found = True
                        break

                if not found:
                    print(
                        f"[grounding] citation not found in any log: {citation}",
                        flush=True,
                    )
                    return False

        return True

    def _validate_citation(self, citation: dict) -> bool:
        """Validate citation structure and types."""
        # Check required fields exist
        required_fields = ["line_start", "line_end", "quote"]
        for field in required_fields:
            if field not in citation:
                return False

        line_start = citation.get("line_start")
        line_end = citation.get("line_end")
        quote = citation.get("quote")

        # Check types
        if not isinstance(line_start, int) or not isinstance(line_end, int):
            return False

        if not isinstance(quote, str):
            return False

        # Check ranges
        if line_start < 1 or line_end < line_start:
            return False

        # Check quote is not empty after stripping
        if not quote.strip():
            return False

        return True

    def _verify_in_log(self, citation: dict, raw_log) -> bool:
        """Try to verify citation in a specific log."""
        line_start = citation["line_start"]
        line_end = citation["line_end"]

        # Check if line numbers are within bounds
        if line_end > len(raw_log.lines):
            return False

        window = raw_log.text_between(line_start, line_end)

        if not window:  # lines are out of range
            return False

        normalized_window = self._normalize(window)
        normalized_quote = self._normalize(citation["quote"])

        # Strict match: quote must appear in the window
        return normalized_quote in normalized_window