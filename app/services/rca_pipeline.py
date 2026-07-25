class RCAPipeline:
    def __init__(self, reasoner, classifier,verifier):
        self.reasoner = reasoner
        self.classifier = classifier
        self.verifier = verifier

    def run(self, evidence, rag_context: list[dict] | None = None) -> dict:
        rca = self.reasoner.generate(
            evidence,
            rag_context=rag_context,
        )
        rca = self.classifier.normalize(rca)

        if self.verifier.verify(rca, evidence):
            return rca

        print("[pipeline] grounding failed, regenerating once...", flush=True)

        rca = self.reasoner.generate(
            evidence,
            rag_context=rag_context,
            strict_grounding=True,
        )

        if self.verifier.verify(rca, evidence):
            return rca

        print("[pipeline] grounding failed again, abstaining", flush=True)

        return {
            "failure_class": "unknown",
            "summary": "Could not produce a grounded RCA.",
            "claims": [],
            "suggested_next_step": "A human should inspect this CI failure manually.",
            "confidence": 0.0,
            "abstained": True,
        }