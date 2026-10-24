class ManualReviewRequired(RuntimeError):
    """An external effect may have happened and must not be repeated blindly."""
