"""Generate text with Ron-1's native checkpoint."""
import argparse
from pathlib import Path
import torch
from ron1_core import ByteTokenizer, Ron1Core, load_config

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--weights",default="core/weights/ron1-native.pt")
    p.add_argument("--prompt",default="أهلاً، أنا رون.")
    p.add_argument("--max-new-tokens",type=int,default=120)
    p.add_argument("--temperature",type=float,default=0.8)
    a=p.parse_args(); path=Path(a.weights)
    if not path.exists(): raise SystemExit(f"Weights missing: {path}. Run python core/initialize.py first.")
    ck=torch.load(path,map_location="cpu",weights_only=False); c=ck.get("config",load_config())
    model=Ron1Core(c); model.load_state_dict(ck["state_dict"]); tok=ByteTokenizer()
    ids=torch.tensor([tok.encode(a.prompt)],dtype=torch.long)
    out=model.generate(ids,max_new_tokens=a.max_new_tokens,temperature=a.temperature)
    print(tok.decode(out[0].tolist()[ids.shape[1]:]))
if __name__=="__main__": main()
