"""
Unit tests for neural text decoding algorithms using synthetic logits and mock models.
Tests Greedy, Beam Search, Top-K, Top-P, and Temperature scaling.
"""
import pytest
import numpy as np
from decoding.decoder import DecoderConfig
from decoding.temperature import apply_temperature
from decoding.greedy import GreedyDecoder
from decoding.beam_search import BeamSearchDecoder
from decoding.top_k_sampling import TopKSampler
from decoding.top_p_sampling import TopPSampler

def test_temperature_scaling():
    logits = np.array([1.0, 2.0, 3.0])
    scaled_hot = apply_temperature(logits, temperature=2.0)
    assert np.allclose(scaled_hot, [0.5, 1.0, 1.5])

    scaled_cold = apply_temperature(logits, temperature=0.5)
    assert np.allclose(scaled_cold, [2.0, 4.0, 6.0])

    # Temperature > 0 strictly required
    with pytest.raises(ValueError):
        apply_temperature(logits, temperature=0.0)
    with pytest.raises(ValueError):
        apply_temperature(logits, temperature=-1.0)

def test_greedy_decoding_deterministic():
    # Model always prefers token 3, then token 4, then EOS (token 5)
    def mock_model(tokens):
        logits = np.zeros(10)
        step = len(tokens)
        if step == 1:
            logits[3] = 10.0
        elif step == 2:
            logits[4] = 10.0
        else:
            logits[5] = 10.0  # EOS
        return logits

    config = DecoderConfig(strategy="greedy", max_new_tokens=5, eos_token_id=5)
    decoder = GreedyDecoder(config)
    result = decoder.generate(mock_model, initial_token_ids=[1])

    assert result.strategy == "greedy"
    assert result.generated_token_ids == [3, 4, 5]

def test_beam_search_scoring():
    # Token 2 has highest initial probability, but sequence [1 -> 3 -> 4] has higher cumulative probability
    def mock_model(tokens):
        logits = np.zeros(10)
        last_tok = tokens[-1]
        if last_tok == 1:
            logits[2] = 2.0  # greedily better
            logits[3] = 1.8
        elif last_tok == 3:
            logits[4] = 5.0  # overall higher score
        elif last_tok == 2:
            logits[5] = 0.5
        return logits

    config = DecoderConfig(strategy="beam_search", beam_width=2, max_new_tokens=2, length_penalty=1.0)
    decoder = BeamSearchDecoder(config)
    result = decoder.generate(mock_model, initial_token_ids=[1])

    assert result.strategy == "beam_search"
    assert len(result.generated_token_ids) == 2
    assert result.generated_token_ids == [3, 4]

def test_top_k_sampling_bounds():
    # Only top 2 tokens should ever be sampled
    def mock_model(tokens):
        logits = np.array([-10.0, -10.0, 5.0, 6.0, -20.0])
        return logits

    config = DecoderConfig(strategy="top_k", top_k=2, max_new_tokens=10, seed=42)
    sampler = TopKSampler(config)
    result = sampler.generate(mock_model, initial_token_ids=[0])

    assert result.strategy == "top_k"
    for tok in result.generated_token_ids:
        assert tok in (2, 3)

def test_top_p_sampling_nucleus():
    # One token has dominant probability
    def mock_model(tokens):
        logits = np.array([0.0, 0.0, 10.0, 0.0])  # token 2 dominates (>99% prob)
        return logits

    config = DecoderConfig(strategy="top_p", top_p=0.8, max_new_tokens=5, seed=42)
    sampler = TopPSampler(config)
    result = sampler.generate(mock_model, initial_token_ids=[0])

    assert result.strategy == "top_p"
    assert result.generated_token_ids == [2, 2, 2, 2, 2]

def test_decoder_config_validation():
    with pytest.raises(ValueError):
        DecoderConfig(temperature=-0.5).validate()
    with pytest.raises(ValueError):
        DecoderConfig(top_k=0).validate()
    with pytest.raises(ValueError):
        DecoderConfig(top_p=1.5).validate()
    with pytest.raises(ValueError):
        DecoderConfig(beam_width=0).validate()
