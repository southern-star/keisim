"""Write keisim/glyphs.py: the 止まれ road marking's three characters as 24 x 24 bitmaps ('#' = paint), rendered once
from Noto Sans CJK JP Bold (SIL Open Font License) so that building a town never needs a font.

usage: uv run python tools/make_tomare.py [FONT]
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = sys.argv[1] if len(sys.argv) > 1 else "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
N, BIG = 24, 192
font = ImageFont.truetype(FONT, int(BIG * 0.92), index=0)        # index 0 of the collection: JP
rows = {}
for ch in "止まれ":
    img = Image.new("L", (BIG, BIG), 0)
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = d.textbbox((0, 0), ch, font=font)
    d.text(((BIG - (x1 - x0)) / 2 - x0, (BIG - (y1 - y0)) / 2 - y0), ch, fill=255, font=font)
    a = np.asarray(img, np.float32).reshape(N, BIG // N, N, BIG // N).mean((1, 3)) / 255.0
    rows[ch] = ["".join("#" if v >= 0.4 else "." for v in r) for r in a]
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "keisim", "glyphs.py")
with open(out, "w") as f:
    f.write('"""The 止まれ road marking (Town.from_graph, Town._make_markings): its three characters as 24 x 24 bitmaps,\n'
            "'#' = paint, row 0 at the top. Rendered from Noto Sans CJK JP Bold (SIL Open Font License) by\n"
            'tools/make_tomare.py."""\n\nTOMARE = [\n')
    for ch in "止まれ":
        f.write(f"    [   # {ch}\n")
        for r in rows[ch]:
            f.write(f'        "{r}",\n')
        f.write("    ],\n")
    f.write("]\n")
print("wrote", out)
for ch in "止まれ":
    print("\n".join(rows[ch]), "\n")
