"""The inference times of two ONNX files in the browser, from the reports of web/tools/pilot_check.mjs (ReconView G2,
docs/recon_view.md): runs/eval/recon/g2_time/{webgpu,wasm}_{public,depth}_{n}.log, taken in turns.

    python tools/recon_time.py runs/eval/recon/g2_time

Prints Markdown: per backend and model the median and the trimmed mean (the middle 80 %) of the model calls' times at the
reports of every run, and the depth model's time over the public one's."""
import argparse
import glob
import os
import re

import numpy as np

ROW = re.compile(r"^\s*[\d.]+\s+\d+\s+\d+\s.*?(\d+) ms\s+(\d+) ms")
JA = {"public": "公開の ONNX（v0.12.0）", "depth": "深度つきの ONNX"}


def times(path):
    return np.array([float(m.group(1)) for m in map(ROW.match, open(path)) if m])


def trimmed(a):
    a = np.sort(a)
    k = len(a) // 10
    return float(a[k:len(a) - k].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    args = ap.parse_args()
    print("| 実行の方法 | モデル | 走行ごとの中央値 [ms] | 走行ごとの平均（中央の 80 %）[ms] | 全部の平均（中央の 80 %）[ms] | 公開版との比 |\n|---|---|---|---|---:|---:|")
    for ep in ("webgpu", "wasm"):
        base = None
        for model in ("public", "depth"):
            runs = [times(p) for p in sorted(glob.glob(os.path.join(args.dir, f"{ep}_{model}_*.log")))]
            runs = [r for r in runs if len(r)]
            if not runs:
                continue
            every = trimmed(np.concatenate(runs))
            base = every if model == "public" else base
            ratio = "–" if model == "public" or not base else f"{100 * (every / base - 1):+.1f} %"
            print(f"| {'WebGPU' if ep == 'webgpu' else 'WASM'} | {JA[model]} | {'、'.join(f'{np.median(r):.0f}' for r in runs)} | "
                  f"{'、'.join(f'{trimmed(r):.1f}' for r in runs)} | {every:.1f} | {ratio} |")
    n = [len(times(p)) for p in glob.glob(os.path.join(args.dir, "*.log"))]
    print(f"\n走行 {len(n)} 本、1 本あたりモデルの呼び出し {min(n)}〜{max(n)} 回ぶんの時間。")


if __name__ == "__main__":
    main()
