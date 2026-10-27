"""
Top-P (Nucleus) Sampling implementation for stochastic text generation.
Selects from the smallest dynamic set of tokens whose cumulative probability exceeds P.
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

class TopPSampler(BaseDecoder):
    """
    Nucleus (Top-P) sampling decoder.
    Dynamically sizes the candidate pool to tokens covering cumulative probability threshold P.
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

        top_p = self.config.top_p

        for step in range(self.config.max_new_tokens):
            logits = model_fn(current_tokens)
            logits = apply_repetition_penalty(logits, generated_tokens, self.config.repetition_penalty)

            # Apply temperature scaling
            scaled_logits = apply_temperature(logits, self.config.temperature)

            # Convert to probabilities
            probs = stable_softmax(scaled_logits)

            # Sort indices by descending probability
            sorted_indices = np.argsort(probs)[::-1]
            sorted_probs = probs[sorted_indices]

            # Cumulative sum
            cumulative_probs = np.cumsum(sorted_probs)

            # Find candidates: keep at least one token, and those up to top_p
            # Elements to remove: cumulative_probs > top_p, shifted right by 1
            indices_to_remove = cumulative_probs > top_p
            indices_to_remove[1:] = indices_to_remove[:-1].copy()
            indices_to_remove[0] = False  # Always preserve highest probability token

            # Zero out removed probabilities
            sorted_probs[indices_to_remove] = 0.0
            sum_p = np.sum(sorted_probs)
            if sum_p > 0:
                sorted_probs /= sum_p
            else:
                sorted_probs[0] = 1.0

            # Sample from the nucleus
            selected_nucleus_idx = rng.choice(len(sorted_indices), p=sorted_probs)
            next_token = int(sorted_indices[selected_nucleus_idx])

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
            strategy="top_p",
            effective_settings={
                "top_p": self.config.top_p,
                "temperature": self.config.temperature,
                "max_new_tokens": self.config.max_new_tokens,
                "repetition_penalty": self.config.repetition_penalty,
                "seed": self.config.seed,
                "eos_token_id": self.config.eos_token_id
            },
            metadata={
                "total_generated": len(generated_tokens),
                "notes": "Top-P adapts dynamically to context entropy: wider pool for flat distributions, narrower pool for peaked ones."
            }
        )
