import torch
import tvm
from torch._export.converter import TS2EPConverter
from exported_program_translator import from_exported_program
import sys


# -----------------------------
# Decoder call mode 확인 함수
# -----------------------------
class JitModuleWrap(torch.nn.Module):
    def __init__(self, jit_mod):
        super().__init__()
        # 명시적으로 서브모듈로 등록 (이 대입으로 이미 등록되긴 함)
        self.jit_mod = jit_mod

    def forward(self, x):
        # ScriptModule은 .forward로 호출해도 되지만, tracing 이슈가 있으니
        # 최종적으로는 wrapped 전체를 trace해서 문제를 피한다.
        return self.jit_mod(x)


class EncNormWrap(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        # 만약 mean/std가 채널별 값이라면 shape 맞춰야 함 (e.g. (3,1,1))
        # 여기선 스칼라로 가정
        x = (x - 0.45) / 0.225
        return self.net(x)


@torch.no_grad()
def detect_decoder_mode(dec, feats):
    try:
        _ = dec(*feats)
        return "varargs"
    except Exception:
        pass

    try:
        _ = dec(feats)
        return "list"
    except Exception as e:
        raise RuntimeError(f"Decoder does not accept varargs nor list: {e}")


# -----------------------------
# Adapter (정상 코드 스타일)
# -----------------------------
class DecoderAdapter(torch.nn.Module):
    def __init__(self, dec, mode):
        super().__init__()
        self.dec = dec
        self.mode = mode

    def forward(self, *feats):
        # varargs: dec(*feats)
        if self.mode == "varargs":
            return self.dec(*feats)
        # list input: dec(list(feats))
        else:
            return self.dec(list(feats))


# -----------------------------
# Encoder Export
# -----------------------------
@torch.no_grad()
def export_encoder_model(model_path, example_inputs):
    # 1) load TorchScript encoder (script module)
    jit_mod = torch.jit.load(model_path).eval()

    # 1a) 내부 Graph 검사: 이미 정규화가 포함되어 있는지 확인
    def jit_module_has_norm(jit_mod):
        try:
            g = jit_mod.inlined_graph
        except Exception:
            g = jit_mod.graph
        s = str(g)
        return ("aten::sub" in s) or ("aten::div" in s) or ("aten::mul" in s)

    has_norm = jit_module_has_norm(jit_mod)
    print(f"[info] jit_mod has_norm = {has_norm}")

    # 2) 래핑하여 nn.Module 계층으로 만든다
    net = JitModuleWrap(jit_mod)

    if has_norm:
        print(
            "[info] Skipping external normalization wrap (encoder already normalizes)."
        )
        target = net.eval()
    else:
        print("[info] Adding normalization wrapper (encoder has no normalization).")
        target = EncNormWrap(net).eval()

    # 핵심: 가중치를 얼려서 최적화 방지
    with torch.jit.optimized_execution(False):
        # BatchNorm folding 등의 최적화 비활성화
        traced = torch.jit.trace(
            target,
            example_inputs,
            check_trace=False,  # 엄격한 체크 비활성화
            strict=False,
        )

    # 4. 가중치 검증 및 복원
    if hasattr(target, "state_dict"):
        original_state = {k: v.clone() for k, v in target.state_dict().items()}

        # trace 후 가중치가 변했는지 확인
        for name, param in traced.named_parameters():
            if name in original_state:
                if not torch.allclose(param, original_state[name], atol=1e-6):
                    print(f"⚠️ Weight changed during trace: {name}")
                    # 원본으로 복원
                    param.data.copy_(original_state[name])

    # 5. TS2EPConverter 사용
    converter = TS2EPConverter(traced, example_inputs, {})
    exported_program = converter.convert()

    # 6. ExportedProgram 검증
    # 변환 후에도 가중치가 유지되는지 확인
    if hasattr(target, "state_dict"):
        for name in original_state:
            ep_weight = None
            # ExportedProgram에서 해당 가중치 찾기
            for param_name, param_value in exported_program.state_dict.items():
                if name in param_name:
                    ep_weight = param_value
                    break

            if ep_weight is not None:
                if not torch.allclose(ep_weight, original_state[name], atol=1e-5):
                    print(f"⚠️ Weight changed in ExportedProgram: {name}")

    return jit_mod, exported_program


# -----------------------------
# Decoder Export (정상과 동일)
# -----------------------------
@torch.no_grad()
def export_decoder_model(enc_mod, decoder_path, dummy):
    # -------------------------
    # 1) Encoder 실행 → feats 획득
    # -------------------------
    enc_feats = enc_mod(dummy)
    sample_feats = []
    for i, f in enumerate(enc_feats):
        if not isinstance(f, torch.Tensor):
            raise RuntimeError(f"Encoder feature {i} is not a Tensor: {type(f)}")
        shape = tuple(f.shape)
        sample_feats.append(torch.randn(*shape))

    # -------------------------
    # 2) Decoder 입력 모드 자동 판별
    # -------------------------
    dec_mod = torch.jit.load(decoder_path, map_location="cpu").eval()
    mode = detect_decoder_mode(dec_mod, sample_feats)

    # -------------------------
    # 3) Adapter 로 decoder 입력 통일
    # -------------------------
    target = DecoderAdapter(dec_mod, mode).eval()

    # ✅ 수정: sample_feats를 tuple로 전달
    with torch.jit.optimized_execution(False):
        traced = torch.jit.trace(
            target,
            tuple(sample_feats),  # ← 여기! dummy가 아니라 sample_feats
            check_trace=True,
            strict=False,
        )

    # 4. 가중치 검증 및 복원
    if hasattr(target, "state_dict"):
        original_state = {k: v.clone() for k, v in target.state_dict().items()}

        for name, param in traced.named_parameters():
            if name in original_state:
                if not torch.allclose(param, original_state[name], atol=1e-6):
                    print(f"⚠️ Weight changed during trace: {name}")
                    param.data.copy_(original_state[name])

    # 5. TS2EPConverter 사용
    # ✅ 수정: 여기도 sample_feats 사용
    converter = TS2EPConverter(traced, tuple(sample_feats), {})
    exported_program = converter.convert()

    # 6. ExportedProgram 검증
    if hasattr(target, "state_dict"):
        for name in original_state:
            ep_weight = None
            for param_name, param_value in exported_program.state_dict.items():
                if name in param_name:
                    ep_weight = param_value
                    break

            if ep_weight is not None:
                if not torch.allclose(ep_weight, original_state[name], atol=1e-5):
                    print(f"⚠️ Weight changed in ExportedProgram: {name}")

    return exported_program


# ======================================================
# 실제 TVM 변환 실행
# ======================================================
input_shape = (1, 3, 192, 640)
dummy = torch.load("dummy_input.pt")

target = "llvm -mtriple=aarch64-linux-gnu"

# 1) Encoder
enc_mod, encoder_program = export_encoder_model(
    "model/mono_640x192_encoder.pt", (dummy,)
)
encoder_relay_mod = from_exported_program(encoder_program)

json_str = tvm.ir.save_json(encoder_relay_mod)

with open("mod_bad_enc.json", "w") as f:
    f.write(json_str)

with tvm.transform.PassContext(opt_level=3):
    ctx = tvm.cpu(0)
    lib = tvm.relay.build(encoder_relay_mod, target=target)

lib.export_library("encoder_deploy_tvm.so", cc="aarch64-linux-gnu-g++")
print("✅ Encoder exported")


# 2) Decoder
decoder_program = export_decoder_model(enc_mod, "model/mono_640x192_decoder.pt", dummy)
decoder_relay_mod = from_exported_program(decoder_program)

with tvm.transform.PassContext(opt_level=3):
    lib = tvm.relay.build(decoder_relay_mod, target=target)

json_str = tvm.ir.save_json(decoder_relay_mod)

with open("mod_bad_dec.json", "w") as f:
    f.write(json_str)


lib.export_library("decoder_deploy_tvm.so", cc="aarch64-linux-gnu-g++")
print("✅ Decoder exported")
