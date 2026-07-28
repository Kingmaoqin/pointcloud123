"""生成图4/图5/图8 的"修改前 / 修改后"对照图，便于代理人核对本次改动。

修改前的图取自代理人反馈稿（技术方案疑问-…-0728.docx）中的嵌入附图；
修改后的图取自本次重画结果。
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/home/xqin5/patent_gap_nbv")
FEEDBACK = ROOT / "技术方案疑问-建筑点云缺口检测及补充扫描视点规划-0728.docx"
NEW_DIR = ROOT / "outputs" / "patent_figures" / "formal_bw"
OUT = ROOT / "交付_20260728_第二轮反馈修订" / "04_本次重画附图"

FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"

# 图号 -> (反馈稿内媒体文件, 重画后文件名, 批注号, 一句话说明)
FIGURES = {
    4: ("word/media/image4.png", "图4_共享边和双法向约束形成表面分块示意图",
        "批注6", "四个三角面改为真正共享边；并入者加阴影、未并入者留白；图文统一为“两项法向约束”"),
    5: ("word/media/image5.png", "图5_点云坐标对齐最近表面点和距离阈值示意图",
        "批注7、8", "补画IFC表面与两侧±τ_d平行虚线；带内实心/带外空心；增加对齐前后对照"),
    8: ("word/media/image8.png", "图8_候选位姿生成示意图",
        "批注9", "候选位姿改为落在三条弧线之上（每层3个）；清理文字重叠；增加图例"),
}


def _font(size: int):
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except OSError:
        return ImageFont.load_default()


def build(num: int) -> Path:
    member, new_stem, comment, note = FIGURES[num]
    with zipfile.ZipFile(FEEDBACK) as z:
        before = Image.open(io.BytesIO(z.read(member))).convert("RGB")
    after = Image.open(NEW_DIR / f"{new_stem}.png").convert("RGB")

    pw = 1500                                  # 每栏宽度
    def fit(im: Image.Image) -> Image.Image:
        h = int(im.height * pw / im.width)
        return im.resize((pw, h), Image.LANCZOS)

    b, a = fit(before), fit(after)
    ph = max(b.height, a.height)
    head, foot, gap, pad = 190, 130, 40, 30
    canvas = Image.new("RGB", (pw * 2 + gap + pad * 2, head + ph + foot), "white")
    d = ImageDraw.Draw(canvas)

    d.text((pad, 26), f"图{num}  修改前后对照（{comment}）", fill="black", font=_font(48))
    for x, im, label in ((pad, b, "修改前（代理人反馈稿）"), (pad + pw + gap, a, "修改后（本次重画）")):
        d.text((x, head - 52), label, fill="black", font=_font(36))
        canvas.paste(im, (x, head))
        d.rectangle([x, head, x + pw, head + im.height], outline="#999999", width=2)

    # 分隔线与说明（单行，避免文字相互覆盖）
    xm = pad + pw + gap // 2
    d.line([(xm, head - 12), (xm, head + ph + 12)], fill="#cccccc", width=3)
    d.text((pad, head + ph + 44), f"本次修改：{note}", fill="black", font=_font(34))

    out = OUT / f"图{num}_修改前后对照.png"
    canvas.save(out)
    print(f"saved -> {out.name}  ({canvas.width}x{canvas.height})")
    return out


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for n in FIGURES:
        build(n)
