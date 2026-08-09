"""把交付用的 Markdown 转成 Word（.docx）。

只处理本项目实际用到的语法：标题、段落、无序/有序列表、表格、代码块、粗体、
行内代码、引用块、水平线。表格按内容宽度自适应，中文字体统一。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor

CN_FONT = "SimSun"          # 正文：宋体
CN_HEAD = "SimHei"          # 标题：黑体
MONO = "Consolas"


def _set_cn(run, font: str) -> None:
    run.font.name = font
    run._element.rPr.rFonts.set(qn("w:eastAsia"), font)


def _emit_inline(par, text: str, base_font: str = CN_FONT) -> None:
    """处理 **粗体** 与 `行内代码`。"""
    for tok in re.split(r"(\*\*.+?\*\*|`[^`]+`)", text):
        if not tok:
            continue
        if tok.startswith("**") and tok.endswith("**"):
            r = par.add_run(tok[2:-2]); r.bold = True; _set_cn(r, base_font)
        elif tok.startswith("`") and tok.endswith("`"):
            r = par.add_run(tok[1:-1]); r.font.name = MONO
            r.font.size = Pt(9); _set_cn(r, MONO)
        else:
            r = par.add_run(tok); _set_cn(r, base_font)


def _table(doc, rows: list[str]) -> None:
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    cells = [c for c in cells if not all(re.fullmatch(r":?-{2,}:?", x or "-") for x in c)]
    if not cells:
        return
    ncol = max(len(r) for r in cells)
    t = doc.add_table(rows=0, cols=ncol)
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(cells):
        cs = t.add_row().cells
        for j in range(ncol):
            txt = row[j] if j < len(row) else ""
            par = cs[j].paragraphs[0]
            par.paragraph_format.space_before = Pt(1)
            par.paragraph_format.space_after = Pt(1)
            _emit_inline(par, txt)
            for r in par.runs:
                r.font.size = Pt(8.5)
                if i == 0:
                    r.bold = True


def convert(md: Path, out: Path) -> None:
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = CN_FONT
    st.font.size = Pt(10.5)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), CN_FONT)

    lines = md.read_text().splitlines()
    i, buf_tbl, in_code, code = 0, [], False, []
    while i < len(lines):
        ln = lines[i]

        if ln.startswith("```"):
            if in_code:
                p = doc.add_paragraph()
                p.paragraph_format.left_indent = Pt(14)
                r = p.add_run("\n".join(code)); r.font.name = MONO
                r.font.size = Pt(8.5); _set_cn(r, MONO)
                r.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
                code, in_code = [], False
            else:
                in_code = True
            i += 1
            continue
        if in_code:
            code.append(ln); i += 1; continue

        if ln.lstrip().startswith("|") and ln.count("|") >= 2:
            buf_tbl.append(ln); i += 1
            if i >= len(lines) or not lines[i].lstrip().startswith("|"):
                _table(doc, buf_tbl); buf_tbl = []
                doc.add_paragraph()
            continue

        if re.fullmatch(r"\s*-{3,}\s*", ln):
            i += 1; continue

        m = re.match(r"^(#{1,6})\s+(.*)$", ln)
        if m:
            lvl, txt = len(m.group(1)), m.group(2)
            h = doc.add_heading(level=min(lvl, 4))
            h.paragraph_format.space_before = Pt(10 if lvl <= 2 else 6)
            h.paragraph_format.space_after = Pt(4)
            for r in h.runs:
                r.text = ""
            _emit_inline(h, txt, CN_HEAD)
            for r in h.runs:
                r.font.color.rgb = RGBColor(0, 0, 0)
                r.font.size = Pt({1: 16, 2: 13.5, 3: 12, 4: 11}.get(lvl, 11))
            i += 1; continue

        if ln.startswith(">"):
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Pt(18)
            _emit_inline(p, ln.lstrip("> ").rstrip())
            for r in p.runs:
                r.italic = True
            i += 1; continue

        def _absorb(idx: int, first: str) -> tuple[str, int]:
            """列表项的续行属于同一项。逐行独立处理会把跨行的 **粗体** 拆开，
            后半截的 ** 就会原样出现在 Word 里。"""
            parts, k = [first], idx + 1
            while (k < len(lines) and lines[k].strip()
                   and re.match(r"^\s{2,}\S", lines[k])
                   and not re.match(r"^\s*([-*]\s|\d+\.\s|\||>|```)", lines[k])):
                parts.append(lines[k].strip()); k += 1
            return "".join(parts), k

        m = re.match(r"^(\s*)[-*]\s+(.*)$", ln)
        if m:
            txt, i = _absorb(i, m.group(2))
            p = doc.add_paragraph(style="List Bullet")
            p.paragraph_format.left_indent = Pt(18 + 14 * (len(m.group(1)) // 2))
            _emit_inline(p, txt)
            continue

        m = re.match(r"^(\s*)(\d+)\.\s+(.*)$", ln)
        if m:
            txt, i = _absorb(i, m.group(3))
            p = doc.add_paragraph(style="List Number")
            p.paragraph_format.left_indent = Pt(18 + 14 * (len(m.group(1)) // 2))
            _emit_inline(p, txt)
            continue

        if not ln.strip():
            i += 1; continue

        # 连续的普通行合并为一段（Markdown 的软换行）
        para = [ln.rstrip()]
        i += 1
        while (i < len(lines) and lines[i].strip()
               and not re.match(r"^(#{1,6}\s|\s*[-*]\s|\s*\d+\.\s|\||>|```)", lines[i])
               and not re.fullmatch(r"\s*-{3,}\s*", lines[i])):
            para.append(lines[i].rstrip()); i += 1
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(4)
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        _emit_inline(p, "".join(para))

    if buf_tbl:
        _table(doc, buf_tbl)
    doc.save(out)


if __name__ == "__main__":
    for src in sys.argv[1:]:
        s = Path(src)
        d = s.with_suffix(".docx")
        convert(s, d)
        print(f"  {d.name}")
