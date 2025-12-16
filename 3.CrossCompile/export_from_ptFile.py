import argparse
import os
import sys
import torch
import tvm
from tvm import relay

# --------- CLI ---------
p = argparse.ArgumentParser()
p.add_argument("--encoder-pt", required=True)
p.add_argument("--decoder-pt", required=True)
p.add_argument("--height", type=int, default=192)
p.add_argument("--width", type=int, default=640)
p.add_argument("--target", type=str, default="llvm -mtriple=aarch64-linux-gnu")
p.add_argument("--outdir", type=str, default="tvm_model")
args = p.parse_args()

ENCODER_PT = args.encoder_pt
DECODER_PT = args.decoder_pt
H, W = args.height, args.width
TARGET = tvm.target.Target(args.target)
OUTDIR = args.outdir
os.makedirs(OUTDIR, exist_ok=True)


def export_tvm_module(lib, name):
    so = os.path.join(OUTDIR, f"{name}.so")
    js = os.path.join(OUTDIR, f"{name}.json")
    pr = os.path.join(OUTDIR, f"{name}.params")
    lib.export_library(so, cc="aarch64-linux-gnu-g++")
    with open(js, "w") as f:
        f.write(lib.get_graph_json())
    with open(pr, "wb") as f:
        f.write(relay.save_param_dict(lib.get_params()))
    print(f"[OK] Exported: {so}, {js}, {pr}")


print("Loading TorchScript models...")
encoder = torch.jit.load(ENCODER_PT, map_location="cpu").eval()
decoder = torch.jit.load(DECODER_PT, map_location="cpu").eval()

dummy = torch.randn(1, 3, H, W)

with torch.no_grad():
    enc_mod = encoder
    print("Running encoder once to get actual feature shapes...")
    enc_out = enc_mod(dummy)

if isinstance(enc_out, torch.Tensor):
    enc_feats = [enc_out]
elif isinstance(enc_out, (list, tuple)):
    enc_feats = list(enc_out)
else:
    raise RuntimeError(f"Unsupported encoder output type: {type(enc_out)}")

print("Converting encoder to Relay IR...")
mod_enc, params_enc = relay.frontend.from_pytorch(enc_mod, [("input_0", dummy.shape)])

print("Building encoder for TVM...")
lib_enc = relay.build(mod_enc, target=TARGET, params=params_enc)
export_tvm_module(lib_enc, "encoder_deploy")


def try_call_decoder(dec, feats):
    with torch.no_grad():
        try:
            _ = dec(*feats)
            return "varargs"
        except Exception:
            pass
        try:
            _ = dec(feats)
            return "list"
        except Exception as e:
            raise RuntimeError(
                f"Decoder doesn't accept either varargs or list. " f"Last error: {e}"
            )


mode = try_call_decoder(decoder, enc_feats)


class DecoderAdapter(torch.nn.Module):
    def __init__(self, dec, mode):
        super().__init__()
        self.dec = dec
        self.mode = mode

    def forward(self, *feats):
        if self.mode == "varargs":
            return self.dec(*feats)
        else:  # "list"
            return self.dec(list(feats))


adapter = DecoderAdapter(decoder, mode).eval()

sample_feats = []
input_list = []
for i, f in enumerate(enc_feats):
    if not isinstance(f, torch.Tensor):
        raise RuntimeError(f"Encoder feature {i} is not a Tensor: {type(f)}")
    shape = tuple(f.shape)
    sample_feats.append(torch.randn(*shape))
    input_list.append((f"input_{i}", shape))


with torch.no_grad():
    traced_adapter = torch.jit.trace(adapter, tuple(sample_feats))
    print("Converting decoder (adapter) to Relay IR...")
    mod_dec, params_dec = relay.frontend.from_pytorch(traced_adapter, input_list)

print("Building decoder for TVM...")
lib_dec = relay.build(mod_dec, target=TARGET, params=params_dec)
export_tvm_module(lib_dec, "decoder_deploy")


print("\n✅ Done. Files are in:", OUTDIR)
print("   - encoder_deploy.so/.json/.params")
print("   - decoder_deploy.so/.json/.params")
