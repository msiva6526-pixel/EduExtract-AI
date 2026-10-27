"""
Generation Interface for EduExtract AI.
Provides unified generation capabilities across:
1. Local NLP Extractive baseline
2. Neural Autoregressive generation with custom decoding strategies
3. Educational logit simulation for NLP Lab experimentation
"""
from typing import Dict, Any, List, Optional
import numpy as np
from decoding.decoder import DecoderConfig, GenerationOutput
from decoding.greedy import GreedyDecoder
from decoding.beam_search import BeamSearchDecoder
from decoding.top_k_sampling import TopKSampler
from decoding.top_p_sampling import TopPSampler
from models.model_loader import ModelLoader
from utils.logging_config import get_logger

logger = get_logger("generation_interface")

class GenerationInterface:
    """Manages text generation using decoding algorithms with hardware awareness."""

    def __init__(self, config: Optional[DecoderConfig] = None):
        self.config = config or DecoderConfig()
        self.model = None
        self.tokenizer = None
        self.device = "cpu"

    def set_config(self, config: DecoderConfig):
        """Update decoding configuration."""
        self.config = config

    def generate_with_decoder(
        self,
        model_fn,
        initial_token_ids: List[int],
        tokenizer = None
    ) -> GenerationOutput:
        """
        Execute generation using the strategy specified in self.config.
        Dispatches to GreedyDecoder, BeamSearchDecoder, TopKSampler, or TopPSampler.
        """
        strategy = self.config.strategy.lower()

        if strategy == "greedy":
            decoder = GreedyDecoder(self.config)
        elif strategy == "beam_search":
            decoder = BeamSearchDecoder(self.config)
        elif strategy == "top_k":
            decoder = TopKSampler(self.config)
        elif strategy == "top_p":
            decoder = TopPSampler(self.config)
        else:
            raise ValueError(f"Unknown decoding strategy: {strategy}")

        return decoder.generate(model_fn, initial_token_ids, tokenizer)

    @staticmethod
    def create_educational_vocab() -> Dict[str, Any]:
        """
        Create a small, interpretable vocabulary for the NLP Lab to demonstrate
        how logits, temperature, Top-K, Top-P, and Beam Search interact token-by-token.
        """
        vocab = [
            "<pad>", "<bos>", "<eos>",
            "Photosynthesis", "is", "a", "process", "where", "plants", "convert",
            "light", "energy", "into", "chemical", "energy.",
            "Water", "and", "carbon", "dioxide", "are", "essential", "reactants.",
            "The", "chloroplast", "captures", "solar", "photons", "efficiently."
        ]
        token2id = {w: i for i, w in enumerate(vocab)}
        id2token = {i: w for i, w in enumerate(vocab)}

        class ToyTokenizer:
            def encode(self, text: str) -> List[int]:
                return [token2id.get(w, 0) for w in text.split()]
            def decode(self, token_ids: List[int]) -> str:
                words = [id2token.get(i, f"[{i}]") for i in token_ids if i not in (0, 1, 2)]
                return " ".join(words)

        # Transition matrix / synthetic logits function based on word associations
        def toy_model_fn(context_tokens: List[int]) -> np.ndarray:
            vocab_size = len(vocab)
            # Base logits with some background noise
            rng = np.random.default_rng(abs(hash(tuple(context_tokens[-3:]))) % (2**31))
            logits = rng.normal(loc=0.0, scale=0.5, size=vocab_size)

            last_token = context_tokens[-1] if context_tokens else 1
            last_word = id2token.get(last_token, "")

            # Educational grammatical priors
            if last_word in ("Photosynthesis", "chloroplast"):
                logits[token2id["is"]] += 4.5
                logits[token2id["captures"]] += 3.5
            elif last_word == "is":
                logits[token2id["a"]] += 5.0
                logits[token2id["essential"]] += 2.5
            elif last_word == "a":
                logits[token2id["process"]] += 5.5
            elif last_word == "process":
                logits[token2id["where"]] += 4.0
            elif last_word == "where":
                logits[token2id["plants"]] += 5.0
            elif last_word == "plants":
                logits[token2id["convert"]] += 5.0
            elif last_word == "convert":
                logits[token2id["light"]] += 5.5
            elif last_word == "light":
                logits[token2id["energy"]] += 6.0
            elif last_word == "energy":
                logits[token2id["into"]] += 4.0
                logits[token2id["<eos>"]] += 2.0
            elif last_word == "into":
                logits[token2id["chemical"]] += 5.5
            elif last_word == "chemical":
                logits[token2id["energy."]] += 6.0
            elif last_word == "energy.":
                logits[token2id["<eos>"]] += 8.0

            return logits

        return {
            "vocab": vocab,
            "token2id": token2id,
            "id2token": id2token,
            "tokenizer": ToyTokenizer(),
            "model_fn": toy_model_fn,
            "eos_id": token2id["<eos>"],
            "bos_id": token2id["<bos>"]
        }
