# utils/errors.py

class PipelineError(Exception):
    """Base exception for the analysis pipeline."""
    pass

class SegmentationError(PipelineError):
    """Raised when wound segmentation fails completely."""
    pass

class TrackingError(PipelineError):
    """Raised when cell tracking fails."""
    pass