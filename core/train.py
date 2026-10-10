"""Train Ron-1 on UTF-8 text using next-byte prediction."""
import argparse, random
from pathlib import Path
import torch
from torch.optim import AdamW
from ron1_core import ByteTokenizer, Ron1Core, load_config, parameter_count

def read_corpus(path):
    files = sorted(path.rglob("*.txt")) if path.is_dir() else [path]
    ids = []
    tok = ByteTokenizer()
    for f in files:
        try: text = f.read_text(encoding="utf-8", errors="replace").strip()
        except OSError: continue
        if text: ids.extend(tok.encode(text, add_bos=True, add_eos=True))
    if len(ids) < 3: raise SystemExit(f"No usable UTF-8 text corpus found at {path}; add .txt files.")
    return torch.tensor(ids, dtype=torch.long)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="UTF-8 .txt file or directory")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--output", default="core/weights/ron1-native.pt")
    p.add_argument("--resume", action="store_true")
    a=p.parse_args(); c=load_config()
    random.seed(c["seed"]); torch.manual_seed(c["seed"])
    model=Ron1Core(c); out=Path(a.output); start=0
    if a.resume and out.exists():
        ck=torch.load(out,map_location="cpu",weights_only=False); model.load_state_dict(ck["state_dict"]); start=int(ck.get("step",0))
    data=read_corpus(Path(a.data)); opt=AdamW(model.parameters(),lr=a.lr,weight_decay=0.01)
    length=min(c["max_seq_len"],data.numel()-1)
    if length < 2: raise SystemExit("Training corpus is too short.")
    model.train()
    for step in range(start,start+a.steps):
        starts=torch.randint(0,data.numel()-length,(a.batch_size,))
        x=torch.stack([data[int(s):int(s)+length] for s in starts])
        y=torch.stack([data[int(s)+1:int(s)+length+1] for s in starts])
        _,loss=model(x,y); opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if (step+1)%50==0 or step==start: print(f"step={step+1} loss={loss.item():.4f}")
    out.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"format":"ron1-native-state-dict-v1","model_name":"Ron-1 Native Core","config":c,"step":start+a.steps,"trained":True,"state_dict":model.state_dict()},out)
    print(f"Saved {out}; steps={start+a.steps}; parameters={parameter_count(model):,}")
if __name__=="__main__": main()
