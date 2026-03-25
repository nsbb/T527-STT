#!/usr/bin/env python3
"""NeMo Conformer ONNX → Acuity 호환 ONNX 변환.

NeMo model.export()로 생성된 ONNX를 Acuity import 가능하도록 수정:
1. Dynamic shape → static [1, 80, 301]
2. length input → 상수 301
3. onnxsim (4462 → 1905 nodes)
4. Pad op의 빈 constant_value → 0.0

Where op은 건드리지 않음! (Acuity가 자체 처리)

Usage:
    python fix_onnx_for_acuity.py input.onnx output.onnx [--frames 301]
"""
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnxsim import simplify
import numpy as np
import argparse
import os

def fix_onnx(input_path, output_path, time_frames=301):
    m = onnx.load(input_path)
    print(f"Input: {len(m.graph.node)} nodes, opset {m.opset_import[0].version}")

    # 1. Static shape [1, 80, TIME_FRAMES]
    for inp in m.graph.input:
        if "audio" in inp.name.lower():
            for d in inp.type.tensor_type.shape.dim:
                d.ClearField("dim_param")
            inp.type.tensor_type.shape.dim[0].dim_value = 1
            inp.type.tensor_type.shape.dim[1].dim_value = 80
            inp.type.tensor_type.shape.dim[2].dim_value = time_frames
            print(f"  Fixed input shape: [1, 80, {time_frames}]")

    # 2. length → constant
    length_init = helper.make_tensor("length", TensorProto.INT64, [1], [time_frames])
    m.graph.initializer.append(length_init)
    inputs = [i for i in m.graph.input if i.name != "length"]
    del m.graph.input[:]
    m.graph.input.extend(inputs)
    print(f"  Replaced length input with constant {time_frames}")

    # 3. onnxsim
    onnx.save(m, "/tmp/_fix_onnx_tmp.onnx")
    m2, ok = simplify(onnx.load("/tmp/_fix_onnx_tmp.onnx"))
    print(f"  onnxsim: {len(m.graph.node)} → {len(m2.graph.node)} nodes, ok={ok}")
    os.remove("/tmp/_fix_onnx_tmp.onnx")

    # 4. Pad fix
    pad_const = "__pad_zero__"
    m2.graph.initializer.append(
        numpy_helper.from_array(np.array(0.0, dtype=np.float32), name=pad_const))
    fixed = 0
    for p in m2.graph.node:
        if p.op_type == "Pad" and len(p.input) >= 3 and p.input[2] == '':
            p.input[2] = pad_const
            fixed += 1
    print(f"  Fixed {fixed} Pad ops (empty constant_value → 0.0)")

    # Stats
    where_count = sum(1 for n in m2.graph.node if n.op_type == "Where")
    print(f"  Where ops remaining: {where_count} (Acuity handles these)")

    onnx.save(m2, output_path)
    print(f"\nOutput: {output_path}")
    print(f"  Nodes: {len(m2.graph.node)}")
    print(f"  Input:  {[(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in m2.graph.input]}")
    print(f"  Output: {[(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in m2.graph.output]}")
    print(f"  Size: {os.path.getsize(output_path)/1e6:.1f}MB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="NeMo exported ONNX")
    parser.add_argument("output", help="Acuity-compatible ONNX")
    parser.add_argument("--frames", type=int, default=301, help="mel time frames (default: 301 ≈ 3s)")
    args = parser.parse_args()
    fix_onnx(args.input, args.output, args.frames)
