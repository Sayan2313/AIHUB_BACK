import json
from LLM.peak.tokenizer_custom import CustomTokenizer
from LLM.peak.architecture.inference import PeakInference
from LLM.peak.architecture.model import PeakModel

def load_model(model_size : str= 'small'):
    # Config
    with open(f"LLM/peak/peak_{model_size}_config.json",'r') as f:
        config = json.load(f)
    # Tokenizer
    tokn = CustomTokenizer(f"weights/peak/{model_size}/vocab_{model_size}.json")
    # Weights Path
    weights = f"weights/peak/{model_size}/best_weights_{model_size}.pth"
    return config,tokn,weights

def inference(data):
    cfg,tokenizer,weight_path = load_model(data.model_size)
    engine = PeakInference(
        PeakModel,
        cfg,
        weight_path,
        tokenizer,
        'cpu'

    )
    return engine.generate(data.text,max_new_tokens=100,temperature=0.7,top_k=40,top_p=0.9,eos_token_id=3)
