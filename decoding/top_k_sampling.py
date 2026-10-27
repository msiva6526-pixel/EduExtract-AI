"""
Top-K Sampling implementation for stochastic text generation.
Filters logits to the top K candidates before applying softmax and sampling.
"""
from typing import List
import numpy as np
from decoding.decoder import (
    BaseDecoder,
    DecoderConfig,
    GenerationOutput,
    stable_softmax,
    apply_repetition_penalty
)
from decoding.temperature import apply_temperature

class TopKSampler(BaseDecoder):
    """
    Top-K sampling decoder.
    Restricts sampling to the top-K highest-probability tokens.
    """

    def generate(
        self,
        model_fn,
        initial_token_ids: List[int],
        tokenizer = None
    ) -> GenerationOutput:
        rng = np.random.default_rng(self.config.seed)
        current_tokens = list(initial_token_ids)
        generated_tokens: List[int] = []

        k = self.config.top_k

        for step in range(self.config.max_new_tokens):
            logits = model_fn(current_tokens)
            logits = apply_repetition_penalty(logits, generated_tokens, self.config.repetition_penalty)

            # Apply temperature scaling
            scaled_logits = apply_temperature(logits, self.config.temperature)

            # Restrict to top-K
            vocab_size = len(scaled_logits)
            effective_k = min(k, vocab_size)

            # Find cutoff threshold
            top_k_indices = np.argsort(scaled_logits)[-effective_k:]
            masked_logits = np.full_like(scaled_logits, -1e9)
            masked_logits[top_k_indices] = scaled_logits[top_k_indices]

            # Probabilities over top-K
            probs = stable_softmax(masked_logits)

            # Sample next token
            next_token = int(rng.choice(len(probs), p=probs))

            generated_tokens.append(next_token)
            current_tokens.append(next_token)

            if self.config.eos_token_id is not None and next_token == self.config.eos_token_id:
                break

        text = ""
        if tokenizer is not None:
            if hasattr(tokenizer, "decode"):
                text = tokenizer.decode(generated_tokens)
            elif callable(tokenizer):
                text = tokenizer(generated_tokens)

        return GenerationOutput(
            generated_text=text,
            generated_token_ids=generated_tokens,
            strategy="top_k",
            effective_settings={
                "top_k": self.config.top_k,
                "temperature": self.config.temperature,
                "max_new_tokens": self.config.max_new_tokens,
                "repetition_penalty": self.config.repetition_penalty,
                "seed": self.config.seed,
                "eos_token_id": self.config.eos_token_id
            },
            metadata={
                "total_generated": len(generated_tokens),
                "notes": "Top-K cuts off the long tail of low-probability words while maintaining diversity."
            }
        )
