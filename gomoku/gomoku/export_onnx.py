"""Export the trained net to ONNX (fp32 + int8 dynamic-quantized) and verify
numerical parity. The int8 ONNX (~300 KB) is the artifact to ship to mobile.

  python3 -m gomoku.export_onnx --ckpt checkpoints/best.pt --out mobile/models
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from .board import Board, encode
from .model import GomokuNet, count_params


def export(ckpt: str, out_dir: str, n: int = 15):
    os.makedirs(out_dir, exist_ok=True)
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    arch = state.get("arch", {}) if isinstance(state, dict) else {}
    net = GomokuNet(n=n, ch=arch.get("ch", 48), blocks=arch.get("blocks", 4))
    net.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
    net.eval()
    print(f"params: {count_params(net)} ({count_params(net) * 4 / 1024:.0f} KB fp32 weights)")

    fp32_path = os.path.join(out_dir, "gomoku_policy_value.onnx")
    torch.onnx.export(
        net, torch.randn(1, 4, n, n), fp32_path,
        opset_version=13, dynamo=False,
        input_names=["planes"], output_names=["policy", "value"],
        dynamic_axes={"planes": {0: "batch"}, "policy": {0: "batch"},
                      "value": {0: "batch"}},
    )
    print(f"exported: {fp32_path} ({os.path.getsize(fp32_path) / 1024:.0f} KB)")

    int8_path = None
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        int8_path = os.path.join(out_dir, "gomoku_policy_value.int8.onnx")
        quantize_dynamic(fp32_path, int8_path, weight_type=QuantType.QInt8)
        print(f"exported: {int8_path} ({os.path.getsize(int8_path) / 1024:.0f} KB)")
    except ImportError as e:
        print(f"(int8 quantization skipped: pip install onnx onnxruntime — {e})")

    # parity check on real mid-game positions
    import onnxruntime as ort
    sess = ort.InferenceSession(fp32_path, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    max_dp = max_dv = 0.0
    x = None
    for _trial in range(8):
        b = Board(n)
        moves = rng.choice([(r, c) for r in range(4, 11) for c in range(4, 11)],
                           size=int(rng.integers(6, 20)), replace=False)
        for (r, c) in moves:
            if b.legal(int(r), int(c)):
                b.play(int(r), int(c))
        x = encode(b)[None]
        with torch.no_grad():
            tp, tv = net(torch.from_numpy(x))
        op, ov = sess.run(None, {"planes": x.astype(np.float32)})
        max_dp = max(max_dp, float(np.abs(op - tp.numpy()).max()))
        max_dv = max(max_dv, float(np.abs(ov - tv.numpy()).max()))
    print(f"parity fp32 vs torch: max |dPolicy logit|={max_dp:.2e}, max |dValue|={max_dv:.2e}")

    if int8_path:
        sess8 = ort.InferenceSession(int8_path, providers=["CPUExecutionProvider"])
        op8, ov8 = sess8.run(None, {"planes": x.astype(np.float32)})
        agree = int(op8[0].argmax()) == int(op[0].argmax())
        print(f"int8 sanity: policy argmax same={agree}, "
              f"value int8={float(ov8[0]):.3f} vs fp32 {float(ov[0]):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/best.pt")
    ap.add_argument("--out", default="mobile/models")
    ap.add_argument("--board", type=int, default=15)
    args = ap.parse_args()
    export(args.ckpt, args.out, args.board)


if __name__ == "__main__":
    main()
