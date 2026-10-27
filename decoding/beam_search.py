"""
Beam Search Decoding implementation for autoregressive text generation.
Maintains top-B candidate hypotheses scored by cumulative log probabilities.
"""
from typing import List, Dict, Any, Tuple
import numpy as np
from decoding.decoder import (
    BaseDecoder,
    DecoderConfig,
    GenerationOutput,
    stable_log_softmax,
    apply_repetition_penalty
)

class BeamSearchDecoder(BaseDecoder):
    """
    Beam search decoder maintaining `beam_width` top candidate hypotheses.
    Uses log-probabilities to prevent underflow and optional length penalty.
    """

    def generate(
        self,
        model_fn,
        initial_token_ids: List[int],
        tokenizer = None
    ) -> GenerationOutput:
        beam_width = self.config.beam_width
        eos_id = self.config.eos_token_id
        length_penalty = self.config.length_penalty

        # Each beam: (tokens_list, cumulative_log_prob)
        beams: List[Tuple[List[int], float]] = [([], 0.0)]
        completed_beams: List[Tuple[List[int], float]] = []

        for step in range(self.config.max_new_tokens):
            all_candidates: List[Tuple[List[int], float]] = []

            for tokens, cum_log_prob in beams:
                current_context = list(initial_token_ids) + tokens
                logits = model_fn(current_context)
                logits = apply_repetition_penalty(logits, tokens, self.config.repetition_penalty)
                log_probs = stable_log_softmax(logits)

                # Consider top 2 * beam_width tokens per beam for expansion
                top_indices = np.argsort(log_probs)[::-1][:beam_width * 2]

                for tok_id in top_indices:
                    new_tokens = tokens + [int(tok_id)]
                    new_score = cum_log_prob + float(log_probs[tok_id])

                    if eos_id is not None and tok_id == eos_id:
                        # Normalize score by length penalty
                        norm_score = new_score / (max(1, len(new_tokens)) ** length_penalty)
                        completed_beams.append((new_tokens, norm_score))
                    else:
                        all_candidates.append((new_tokens, new_score))

            if not all_candidates:
                break

            # Sort all candidates by cumulative score descending and keep top beam_width
            all_candidates.sort(key=lambda x: x[1], reverse=True)
            beams = all_candidates[:beam_width]

            # Early stopping if completed beams dominate
            if len(completed_beams) >= beam_width:
                break

        # Select best sequence among completed or remaining active beams
        candidates_to_score = completed_beams if completed_beams else [
            (b[0], b[1] / (max(1, len(b[0])) ** length_penalty)) for b in beams
        ]
        candidates_to_score.sort(key=lambda x: x[1], reverse=True)
        best_tokens, best_score = candidates_to_score[0]

        text = ""
        if tokenizer is not None:
            if hasattr(tokenizer, "decode"):
                text = tokenizer.decode(best_tokens)
            elif callable(tokenizer):
                text = tokenizer(best_tokens)

        return GenerationOutput(
            generated_text=text,
            generated_token_ids=best_tokens,
            strategy="beam_search",
            effective_settings={
                "beam_width": beam_width,
                "length_penalty": length_penalty,
                "max_new_tokens": self.config.max_new_tokens,
                "repetition_penalty": self.config.repetition_penalty,
                "eos_token_id": eos_id
            },
            metadata={
                "total_generated": len(best_tokens),
                "best_score": round(best_score, 4),
                "completed_hypotheses": len(completed_beams),
                "notes": "Beam search explores multiple hypotheses but may favor generic phrasing."
            }
        )
