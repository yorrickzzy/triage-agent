VALID_FAILURE_CLASSES = {
    "lint",
    "format",
    "test",
    "build",
    "dependency",
    "flaky",
}


class FailureClassifier:

    def normalize(self, rca: dict) -> dict:

        failure_class = (
            rca.get("failure_class", "")
            .strip()
            .lower()
        )

        if failure_class not in VALID_FAILURE_CLASSES:
            failure_class = "unknown"

        rca["failure_class"] = failure_class

        return rca