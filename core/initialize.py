"""Create Ron-1's own initial weights without downloading a checkpoint."""
import argparse, random
from pathlib import Path
import torch
from ron1_core import Ron1Core, load_config, parameter_count

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="core/weights/ron1-native.pt")
    args = p.parse_args()
    c = load_config()
    random.seed(c["seed"]); torch.manual_seed(c["seed"])
    model = Ron1Core(c)
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format":"ron1-native-state-dict-v1","model_name":"Ron-1 Native Core","config":c,"step":0,"trained":False,"state_dict":model.state_dict()}, out)
    print(f"Initialized {out} ({out.stat().st_size:,} bytes); trainable parameters={parameter_count(model):,}")
    print("Fresh random initialization only; this checkpoint has not learned language yet.")
if __name__ == "__main__": main()
