"""
Model Loader module for optional Generative AI backends in EduExtract AI.
Safely detects PyTorch, Transformers, CPU/CUDA environments without failing if missing.
"""
from typing import Dict, Any, Optional
from utils.logging_config import get_logger
from config import LOCAL_MODEL_NAME

logger = get_logger("model_loader")

class ModelLoader:
    """Manages detection and safe loading of neural language models."""

    @staticmethod
    def get_hardware_info() -> Dict[str, Any]:
        """Detect available hardware capabilities (CUDA/CPU/Apple Silicon)."""
        info = {
            "torch_available": False,
            "cuda_available": False,
            "device": "cpu",
            "transformers_available": False
        }
        try:
            import torch
            info["torch_available"] = True
            if torch.cuda.is_available():
                info["cuda_available"] = True
                info["device"] = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                info["device"] = "mps"
        except ImportError:
            pass

        try:
            import transformers
            info["transformers_available"] = True
        except ImportError:
            pass

        return info

    @staticmethod
    def load_hf_model_and_tokenizer(model_name: str = LOCAL_MODEL_NAME):
        """
        Attempt to load a local HuggingFace causal LM or seq2seq model and tokenizer.
        Returns (model, tokenizer, device, status_msg)
        """
        hw = ModelLoader.get_hardware_info()
        if not hw["transformers_available"] or not hw["torch_available"]:
            return None, None, "cpu", "Transformers or PyTorch not installed in environment."

        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
            import torch

            logger.info("Loading model %s on %s...", model_name, hw["device"])
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            try:
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            except Exception:
                model = AutoModelForCausalLM.from_pretrained(model_name)

            device = torch.device(hw["device"])
            model.to(device)
            model.eval()
            return model, tokenizer, hw["device"], "Model loaded successfully."
        except Exception as e:
            logger.warning("Failed to load Hugging Face model %s: %s", model_name, e)
            return None, None, "cpu", f"Failed to load model: {str(e)}"
