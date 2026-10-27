"""
Greedy Decoding implementation for autoregressive text generation.
Selects the argmax token at each step deterministically.
"""
from typing import List
import numpy as np
from decoding.decoder import BaseDecoder, DecoderConfig, GenerationOutput, apply_repetition_penalty

class GreedyDecoder(BaseDecoder):
    """
    Deterministic greedy search decoder.
    At each step: token = argmax(logits).
    """

    def generate(
        self,
        model_fn,
        initial_token_ids: List[int],
        tokenizer = None
    ) -> GenerationOutput:
        current_tokens = list(initial_token_ids)
        generated_tokens: List[int] = []
        step_logits: List[float] = []

        for step in range(self.config.max_new_tokens):
            logits = model_fn(current_tokens)
            logits = apply_repetition_penalty(logits, generated_tokens, self.config.repetition_penalty)

            # Greedy selection: highest logit
            next_token = int(np.argmax(logits))
            step_logits.append(float(logits[next_token]))

            generated_tokens.append(next_token)
            current_tokens.append(next_token)

            if self.config.eos_token_id is not None and next_token == self.config.eos_token_id:
                break

        # Decode text if tokenizer provided
        text = ""
        if tokenizer is not None:
            if hasattr(tokenizer, "decode"):
                text = tokenizer.decode(generated_tokens)
            elif callable(tokenizer):
                text = tokenizer(generated_tokens)

        return GenerationOutput(
            generated_text=text,
            generated_token_ids=generated_tokens,
            strategy="greedy",
            effective_settings={
                "max_new_tokens": self.config.max_new_tokens,
                "repetition_penalty": self.config.repetition_penalty,
                "eos_token_id": self.config.eos_token_id
            },
            metadata={
                "total_generated": len(generated_tokens),
                "deterministic": True,
                "notes": "Greedy search is fast and deterministic, but can produce repetitive patterns."
            }
        )
