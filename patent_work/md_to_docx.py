"""把答复文件（Markdown）转为代理人惯用的 docx。

markdown -> HTML -> docx（LibreOffice），保留标题层级、表格与加粗。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import markdown

SOFFICE = "/home/xqin5/PPTforTreatment/tools/lo_root/opt/libreoffice26.2/program/soffice"

CSS = """
body { font-family: "Noto Sans CJK SC", "SimSun", sans-serif; font-size: 11pt; line-height: 1.6; }
h1 { font-size: 18pt; }
h2 { font-size: 14pt; margin-top: 18pt; }
h3 { font-size: 12pt; margin-top: 14pt; }
table { border-collapse: collapse; width: 100%; }
th, td { border: 1px solid #666; padding: 4pt 6pt; font-size: 10pt; vertical-align: top; }
th { background: #eeeeee; }
code { font-family: Consolas, monospace; }
hr { border: none; border-top: 1px solid #bbbbbb; }
"""


def convert(md_path: Path, out_dir: Path) -> Path:
    html_body = markdown.markdown(
        md_path.read_text(encoding="utf-8"),
        extensions=["tables", "sane_lists"],
    )
    html = (f'<html><head><meta charset="utf-8"><style>{CSS}</style></head>'
            f"<body>{html_body}</body></html>")
    tmp_html = out_dir / (md_path.stem + ".html")
    tmp_html.write_text(html, encoding="utf-8")

    subprocess.run(
        [SOFFICE, "--headless", "--norestore", "--convert-to", "docx:MS Word 2007 XML",
         "--outdir", str(out_dir), str(tmp_html)],
        check=True, capture_output=True, timeout=180,
    )
    tmp_html.unlink()
    out = out_dir / (md_path.stem + ".docx")
    if not out.exists():
        raise RuntimeError(f"转换失败：未生成 {out}")
    return out


if __name__ == "__main__":
    src = Path(sys.argv[1])
    dst_dir = Path(sys.argv[2])
    dst_dir.mkdir(parents=True, exist_ok=True)
    print("saved ->", convert(src, dst_dir))
