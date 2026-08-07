"""
exceptions.py
-------------
Custom exception hierarchy for the loan eligibility pipeline.
"""


class LoanPipelineError(Exception):
    """Base class for all project-specific exceptions."""


class DataValidationError(LoanPipelineError):
    """Raised when input data fails schema/quality checks."""


class DataLoadError(LoanPipelineError):
    """Raised when the raw dataset cannot be located or read."""


class ModelNotFoundError(LoanPipelineError):
    """Raised when a trained model artifact is expected but missing."""


class ModelTrainingError(LoanPipelineError):
    """Raised when model training fails."""


class PredictionError(LoanPipelineError):
    """Raised when inference fails on otherwise-valid input."""
