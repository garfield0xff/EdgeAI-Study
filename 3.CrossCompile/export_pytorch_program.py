import torch
import tvm
from torch._export.converter import TS2EPConverter
from exported_program_translator import from_exported_program


@torch.no_grad()
def export_decoder_model(encoder_path, decoder_path, dummy):
    enc_mod = torch.jit.load(encoder_path, map_location="cpu").eval()
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

    # 입력 텐서들의 정확한 shape를 encoder 실제 출력으로부터 생성
    print(f'length of enc feats: {len(enc_feats)}')
    sample_feats = []
    for i, f in enumerate(enc_feats):
        print(f'f dtype: {f.dtype}, enc feats shape: {f.shape}')
        if not isinstance(f, torch.Tensor):
            raise RuntimeError(f"Encoder feature {i} is not a Tensor: {type(f)}")
        # 안전상 dtype/shape 유지. dtype은 float32 가 보통.
        shape = tuple(f.shape)
        sample_feats.append(torch.randn(*shape, dtype=f.dtype))

    dec_mod = torch.jit.load(decoder_path, map_location="cpu").eval()
    converter = TS2EPConverter(dec_mod, tuple(sample_feats), {})
    exported_program = converter.convert()
    return exported_program


@torch.no_grad()
def export_encoder_model(model_path, example_inputs):
    model = torch.jit.load(model_path).eval()
    model.eval()
    converter = TS2EPConverter(model, example_inputs, {})
    exported_program = converter.convert()
    return exported_program


# 더미 입력 정의
input_shape = (1, 3, 192, 640)
input_data = torch.randn(input_shape)
target = "llvm -mtriple=aarch64-linux-gnu"  # or "cuda" for GPU
encoder_program = export_encoder_model("model/mono_640x192_encoder.pt", (input_data, ))
encoder_relay_mod = from_exported_program(encoder_program)

with tvm.transform.PassContext(opt_level=3):
    lib = tvm.relay.build(encoder_relay_mod, target=target)


encoder_lib_path = "encoder_deploy_tvm.so"
lib.export_library(encoder_lib_path, cc="aarch64-linux-gnu-g++")
print(f"✅ TVM Encoder shared library exported: {encoder_lib_path}")

decoder_program = export_decoder_model("model/mono_640x192_encoder.pt", "model/mono_640x192_decoder.pt", input_data)
decoder_relay_mod = from_exported_program(decoder_program)
print(f'decoder relay mod: {decoder_relay_mod}')

with tvm.transform.PassContext(
    opt_level=3,
    config={
        "relay.ToMixedPrecision.keep_orig_output_dtype": True
    }
):
    lib = tvm.relay.build(decoder_relay_mod, target=target)


decoder_lib_path = "decoder_deploy_tvm.so"
lib.export_library(decoder_lib_path, cc="aarch64-linux-gnu-g++")
print(f"✅ TVM Decoder shared library exported: {decoder_lib_path}")
