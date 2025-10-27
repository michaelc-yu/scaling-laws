
import yaml
from model.decoder_only import DecoderOnlyLM

def build_model(config_path, model_size):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)["models"][model_size]
    return DecoderOnlyLM(**cfg)

