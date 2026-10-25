import re

TIMESTAMP_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}T.*?Z\s+"
)
# Strips "123 | " line-number prefix the model copies from numbered log output.
LINE_NUMBER_PREFIX = re.compile(r"^\d+\s*\|\s*")
ANSI_ESCAPE_PATTERN = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
ESCAPED_UNICODE_CONTROL_PATTERN = re.compile(r"\\u00[0-9a-fA-F]{2}")

class GroundingVerifier:

    def _normalize(self, text: str) -> str:
        lines = []

        for line in text.splitlines():
            line = ANSI_ESCAPE_PATTERN.sub("", line)
            line = ESCAPED_UNICODE_CONTROL_PATTERN.sub("", line)
            line = CONTROL_CHAR_PATTERN.sub("", line)
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
        return not self.validation_errors(rca, evidence)

    def validation_errors(self, rca: dict, evidence) -> list[str]:
        if rca.get("abstained") is True and not rca.get("claims"):
            return []
        claims = rca.get("claims")
        if not isinstance(claims, list) or not claims:
            return ["A non-abstained RCA must include at least one claim with log citations"]
        errors = []
        for index, claim in enumerate(claims):
            if not isinstance(claim, dict) or not isinstance(claim.get("citations"), list) or not claim["citations"]:
                errors.append(f"Claim {index + 1} needs log citations")
                continue
            for citation in claim.get("citations", []):
                # First validate citation structure
                if not isinstance(citation, dict) or not self._validate_citation(citation):
                    print(
                        f"[grounding] citation validation failed: {citation}",
                        flush=True,
                    )
                    errors.append(f"Claim {index + 1}: citation must have valid line_start, line_end and nonempty quote")
                    continue

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
                    errors.append(f"Claim {index + 1}: quote does not match the snapshot at lines {citation['line_start']}..{citation['line_end']}")

        return errors

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
