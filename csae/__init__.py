"""SAE++: Cascaded Sparse Autoencoders for multi-level concept discovery in MLLMs.

The package is named `csae` (cascaded SAE) because `sae++` is not a valid
Python identifier.
"""

from csae.model import BatchTopKSAE, TwoLevelBatchTopKSAE, TwoLevelBatchTopKTrainer

__all__ = ["BatchTopKSAE", "TwoLevelBatchTopKSAE", "TwoLevelBatchTopKTrainer"]
