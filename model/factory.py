
import yaml
from model.decoder_only import DecoderOnlyLM

def build_model(model_cfg, model_size):
    print(f"[DBG] Building model model_cfg: {model_cfg}...")
    return DecoderOnlyLM(**model_cfg)

