"""
Decoding package for neural text generation.
Implements Greedy decoding, Beam Search, Top-K sampling, Top-P (nucleus) sampling,
and Temperature scaling for autoregressive language models.
"""
from decoding.decoder import DecoderConfig, GenerationOutput, BaseDecoder
from decoding.temperature import apply_temperature
from decoding.greedy import GreedyDecoder
from decoding.beam_search import BeamSearchDecoder
from decoding.top_k_sampling import TopKSampler
from decoding.top_p_sampling import TopPSampler

__all__ = [
    "DecoderConfig",
    "GenerationOutput",
    "BaseDecoder",
    "apply_temperature",
    "GreedyDecoder",
    "BeamSearchDecoder",
    "TopKSampler",
    "TopPSampler"
]
