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

# --------- 1) Encoder: TorchScript -> Relay ---------
dummy = torch.randn(1, 3, H, W)
with torch.no_grad():
    # TorchScript 모듈은 trace가 필요 없으면 경고를 냅니다. 그대로 사용해도 OK.
    enc_mod = encoder
    print("Running encoder once to get actual feature shapes...")
    enc_out = enc_mod(dummy)

# enc_out 이 Tensor 하나가 아니라 list/tuple 여야 함 (MonoDepth2)
if isinstance(enc_out, torch.Tensor):
    # 안전장치: 단일 텐서인 경우에도 리스트로 통일
    enc_feats = [enc_out]
elif isinstance(enc_out, (list, tuple)):
    enc_feats = list(enc_out)
else:
    raise RuntimeError(f"Unsupported encoder output type: {type(enc_out)}")

# from_pytorch는 ScriptModule을 직접 받아도 됩니다.
print("Converting encoder to Relay IR...")
mod_enc, params_enc = relay.frontend.from_pytorch(enc_mod, [("input_0", dummy.shape)])

print("Building encoder for TVM...")
lib_enc = relay.build(mod_enc, target=TARGET, params=params_enc)
export_tvm_module(lib_enc, "encoder_deploy")


# --------- 2) Decoder: 입력 시그니처 자동 판별 -> Relay ---------
# decoder가 (*feats)로 받는지, (feats) 리스트 하나로 받는지 확인
def try_call_decoder(dec, feats):
    """returns mode: 'varargs' if dec(*feats) works, 'list' if dec(feats) works."""
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
print(f"Decoder call mode detected: {mode}")


# TVM의 from_pytorch는 텐서 입력들만 명시하는 게 가장 안전합니다.
# decoder가 리스트 한 개를 받는다면, 어댑터로 (*tensors) -> list 로 바꿔줍니다.
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

# 입력 텐서들의 정확한 shape를 encoder 실제 출력으로부터 생성
sample_feats = []
input_list = []
for i, f in enumerate(enc_feats):
    if not isinstance(f, torch.Tensor):
        raise RuntimeError(f"Encoder feature {i} is not a Tensor: {type(f)}")
    # 안전상 dtype/shape 유지. dtype은 float32 가 보통.
    shape = tuple(f.shape)
    sample_feats.append(torch.randn(*shape))
    input_list.append((f"input_{i}", shape))

# 이제 어댑터를 “N개의 텐서 입력”으로 trace해서 TVM 변환
with torch.no_grad():
    # trace 입력은 튜플로 넘겨야 (*feats) 호출 경로가 캡처됩니다.
    traced_adapter = torch.jit.trace(adapter, tuple(sample_feats))
    # from_pytorch에 input_list를 N개의 텐서로 명시
    print("Converting decoder (adapter) to Relay IR...")
    mod_dec, params_dec = relay.frontend.from_pytorch(traced_adapter, input_list)

print("Building decoder for TVM...")
lib_dec = relay.build(mod_dec, target=TARGET, params=params_dec)
export_tvm_module(lib_dec, "decoder_deploy")

print("\n✅ Done. Files are in:", OUTDIR)
print("   - encoder_deploy.so/.json/.params")
print("   - decoder_deploy.so/.json/.params")
