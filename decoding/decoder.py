"""
Shared Decoder interface and data structures for neural autoregressive generation.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np

@dataclass
class DecoderConfig:
    """Configuration settings for autoregressive text decoding."""
    strategy: str = "greedy"  # "greedy", "beam_search", "top_k", "top_p"
    max_new_tokens: int = 50
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    beam_width: int = 4
    length_penalty: float = 1.0
    repetition_penalty: float = 1.0
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None
    seed: Optional[int] = None

    def validate(self) -> None:
        """Validate decoding hyperparameters."""
        if self.max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be >= 1, got {self.max_new_tokens}")
        if self.temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {self.temperature}")
        if self.beam_width < 1:
            raise ValueError(f"beam_width must be >= 1, got {self.beam_width}")
        if self.top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {self.top_k}")
        if not (0.0 < self.top_p <= 1.0):
            raise ValueError(f"top_p must be in (0.0, 1.0], got {self.top_p}")
        if self.repetition_penalty < 1.0:
            raise ValueError(f"repetition_penalty must be >= 1.0, got {self.repetition_penalty}")

@dataclass
class GenerationOutput:
    """Standardized output result from a decoding generation run."""
    generated_text: str
    generated_token_ids: List[int]
    strategy: str
    effective_settings: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

def stable_softmax(logits: np.ndarray) -> np.ndarray:
    """Compute numerically stable softmax over 1D numpy array."""
    logits = np.asarray(logits, dtype=np.float64)
    # Subtract max for numerical stability
    shifted = logits - np.max(logits)
    exp_vals = np.exp(shifted)
    sum_exp = np.sum(exp_vals)
    if sum_exp == 0 or np.isnan(sum_exp):
        # Uniform fallback
        return np.ones_like(logits) / len(logits)
    return exp_vals / sum_exp

def stable_log_softmax(logits: np.ndarray) -> np.ndarray:
    """Compute numerically stable log-softmax: x_i - max(x) - log(sum(exp(x - max(x))))."""
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits)
    log_sum_exp = np.log(np.sum(np.exp(shifted)) + 1e-12)
    return shifted - log_sum_exp

def apply_repetition_penalty(
    logits: np.ndarray,
    generated_tokens: List[int],
    penalty: float
) -> np.ndarray:
    """Discount logits of tokens that have already been generated."""
    if penalty <= 1.0 or not generated_tokens:
        return logits
    penalized = logits.copy()
    unique_tokens = set(generated_tokens)
    for token_id in unique_tokens:
        if token_id < len(penalized):
            # If logit > 0 divide by penalty, if logit < 0 multiply by penalty
            if penalized[token_id] > 0:
                penalized[token_id] /= penalty
            else:
                penalized[token_id] *= penalty
    return penalized

class BaseDecoder(ABC):
    """Abstract base class for all token-level decoding strategies."""

    def __init__(self, config: Optional[DecoderConfig] = None):
        self.config = config or DecoderConfig()
        self.config.validate()

    @abstractmethod
    def generate(
        self,
        model_fn,
        initial_token_ids: List[int],
        tokenizer = None
    ) -> GenerationOutput:
        """
        Execute generation step-by-step.
        model_fn is a callable: (List[int]) -> np.ndarray returning 1D next-token logits.
        """
        pass
