"""A sheet of ReconView pictures for docs/recon_view.md from web/tools/recon_check.mjs --shots:

    python tools/recon_sheet.py OUT.jpg SHOTS_DIR TOWN:SEC:CAPTION [TOWN:SEC:CAPTION ...] [--width 640] [--model]

One row per scene: the town itself from the chase camera, the reconstruction from the same camera, and the map of the
measurement (green: drawn right, orange: through, red: phantom, blue: seen by a camera but not drawn, grey: not seen)."""
import argparse
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONTS = ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
         "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]


def font(size):
    for f in FONTS:
        if os.path.exists(f):
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("shots")
    ap.add_argument("scenes", nargs="+")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--model", action="store_true", help="shots of ?recon=model: the oracle's reconstruction of the same frame beside the model's")
    ap.add_argument("--accum", action="store_true", help="shots with keyframes: the current frame alone beside the view with them")
    args = ap.parse_args()
    kinds = [("town", cv2.INTER_AREA)] + ([("oracle", cv2.INTER_AREA)] if args.model else []) + ([("current", cv2.INTER_AREA)] if args.accum else []) + [("recon", cv2.INTER_AREA), ("map", cv2.INTER_NEAREST)]
    heads = ["実際の街（追跡カメラ）"] + (["ReconView（深度は正解）", "ReconView（深度はモデル）", "計測の内訳（モデルの深度）"] if args.model
                                  else ["ReconView（同じ追跡カメラ、深度は正解）", "計測の内訳"])
    if args.accum:
        heads = ["実際の街（追跡カメラ）", "A: 今のフレームだけ", "A+: キーフレームを足す", "計測の内訳（A+）"]
    N = len(kinds)
    W, H, bar = args.width, args.width // 2, 34
    rows = []
    for sc in args.scenes:
        town, sec, caption = sc.split(":", 2)
        tag = os.path.join(args.shots, f"{town}_t{int(sec):03d}")
        tiles = []
        for kind, interp in kinds:
            im = cv2.imread(f"{tag}_{kind}.png")
            if im is None:
                raise SystemExit(f"missing {tag}_{kind}.png")
            tiles.append(cv2.resize(im, (W, H), interpolation=interp))
        strip = Image.new("RGB", (N * W, bar), (20, 22, 34))
        d = ImageDraw.Draw(strip)
        d.text((10, 5), f"{caption}（{town}、{sec} 秒）", font=font(20), fill=(235, 235, 245))
        d.text(((N - 1) * W + 10, 8), "計測: 緑 正しい / 橙 透け / 赤 実在しない面 / 青 見えたのに描けず / 灰 見えない", font=font(15), fill=(170, 172, 190))
        rows.append(np.concatenate([np.asarray(strip)[:, :, ::-1], np.concatenate(tiles, 1)], 0))
    head = Image.new("RGB", (N * W, 30), (12, 13, 20))
    d = ImageDraw.Draw(head)
    for i, name in enumerate(heads):
        d.text((i * W + 10, 4), name, font=font(18), fill=(200, 205, 230))
    sheet = np.concatenate([np.asarray(head)[:, :, ::-1]] + rows, 0)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    cv2.imwrite(args.out, sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(args.out, sheet.shape)


if __name__ == "__main__":
    main()
