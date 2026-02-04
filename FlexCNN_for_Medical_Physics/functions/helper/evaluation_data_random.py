"""
DEPRECATED: This module has been refactored into cross_validation.py

Cross-validation (batch loading, evaluation, and Ray Tune reporting) has been moved to:
  FlexCNN_for_Medical_Physics.functions.cross_validation

For backward compatibility, imports are re-exported below.
"""

# Re-export for backward compatibility
from FlexCNN_for_Medical_Physics.functions.cross_validation import (
    load_validation_batch,
    load_qa_batch,
    evaluate_val,
    evaluate_val_frozen,
    evaluate_qa,
    evaluate_qa_frozen,
)

__all__ = [
    'load_validation_batch',
    'load_qa_batch',
    'evaluate_val',
    'evaluate_val_frozen',
    'evaluate_qa',
    'evaluate_qa_frozen',
]

"""
DEPRECATED: This module has been refactored into cross_validation.py

Cross-validation (batch loading, evaluation, and Ray Tune reporting) has been moved to:
  FlexCNN_for_Medical_Physics.functions.cross_validation

For backward compatibility, imports are re-exported below.
"""

# Re-export for backward compatibility
from FlexCNN_for_Medical_Physics.functions.cross_validation import (
    load_validation_batch,
    load_qa_batch,
    evaluate_val,
    evaluate_val_frozen,
    evaluate_qa,
    evaluate_qa_frozen,
)

__all__ = [
    'load_validation_batch',
    'load_qa_batch',
    'evaluate_val',
    'evaluate_val_frozen',
    'evaluate_qa',
    'evaluate_qa_frozen',
]
