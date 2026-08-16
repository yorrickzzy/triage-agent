from app.core.telemetry import record_exception, set_span_attributes, tracer


class RCAPipeline:
    def __init__(self, reasoner, classifier,verifier):
        self.reasoner = reasoner
        self.classifier = classifier
        self.verifier = verifier

    def run(self, evidence, rag_context: list[dict] | None = None) -> dict:
        if tracer is None:
            return self._run_impl(evidence, rag_context=rag_context)

        with tracer.start_as_current_span("grounding.verify") as span:
            try:
                set_span_attributes(
                    span,
                    run_id=evidence.run_id,
                    workflow_name=evidence.workflow_name,
                    branch=evidence.branch,
                )
                result = self._run_impl(evidence, rag_context=rag_context)
                set_span_attributes(
                    span,
                    failure_class=result.get("failure_class"),
                    abstained=bool(result.get("abstained", False)),
                    confidence=result.get("confidence"),
                )
                return result
            except Exception as exc:
                record_exception(span, exc)
                raise

    def _run_impl(self, evidence, rag_context: list[dict] | None = None) -> dict:
        with tracer.start_as_current_span("agent.reasoning") as span:
            first_rca = self.reasoner.generate(
                evidence,
                rag_context=rag_context,
            )
            set_span_attributes(
                span,
                run_id=evidence.run_id,
                workflow_name=evidence.workflow_name,
                strict_grounding=False,
            )
            first_rca = self.classifier.normalize(first_rca)

        if self.verifier.verify(first_rca, evidence):
            return first_rca

        print("[pipeline] grounding failed, regenerating once...", flush=True)

        with tracer.start_as_current_span("agent.reasoning") as span:
            second_rca = self.reasoner.generate(
                evidence,
                rag_context=rag_context,
                strict_grounding=True,
            )
            set_span_attributes(
                span,
                run_id=evidence.run_id,
                workflow_name=evidence.workflow_name,
                strict_grounding=True,
            )
            second_rca = self.classifier.normalize(second_rca)

        if self.verifier.verify(second_rca, evidence):
            return second_rca

        print("[pipeline] grounding failed again, abstaining", flush=True)

        reasoning_trace = []
        if isinstance(first_rca.get("reasoning_trace"), list):
            reasoning_trace.extend(first_rca["reasoning_trace"])
        reasoning_trace.append(
            {
                "event": "pipeline_grounding_failed",
                "pass": 1,
            }
        )

        if isinstance(second_rca.get("reasoning_trace"), list):
            reasoning_trace.extend(second_rca["reasoning_trace"])
        reasoning_trace.append(
            {
                "event": "pipeline_grounding_failed",
                "pass": 2,
                "action": "abstain",
            }
        )

        return {
            "failure_class": "unknown",
            "summary": "Could not produce a grounded RCA.",
            "claims": [],
            "suggested_next_step": "A human should inspect this CI failure manually.",
            "confidence": 0.0,
            "abstained": True,
            "reasoning_trace": reasoning_trace,
        }