"""
Temperature scaling module for logit probability distributions.
"""
import numpy as np

def apply_temperature(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Scale next-token logits by temperature: scaled_logits = logits / temperature.

    Parameters:
        logits: 1D array of unnormalized token logits.
        temperature: Positive float (> 0).
                     - T < 1.0: Sharper distribution, favors high-probability tokens.
                     - T = 1.0: Original distribution unchanged.
                     - T > 1.0: Flatter distribution, increases diversity/randomness.

    Note:
        Positive temperature scaling strictly preserves rank ordering; it does not change
        argmax(logits), so greedy decoding is invariant to temperature.
    """
    if temperature <= 0.0:
        raise ValueError(f"Temperature must be strictly greater than 0, got {temperature}")

    logits = np.asarray(logits, dtype=np.float64)
    # Scaled logits
    return logits / float(temperature)
