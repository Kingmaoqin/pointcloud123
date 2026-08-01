"""修正 md→docx 转换后的两处表格排版问题（2026-07-31 反馈 三·1、三·2）。

1. 宽表超出页面：把含宽表的文档整篇改为横向 A4，并按列数重新分配列宽，
   使最右侧列（如"性质"）不再被裁掉。
2. 表格跨页断裂：关闭所有表格行的"允许跨页断行"，并把表头设为每页重复，
   避免出现"上一页表格末尾断开、下一页只剩孤立一两个字"的情况。

用法：python3 fix_docx_layout.py <docx> [--landscape]
"""

from __future__ import annotations

import sys
from pathlib import Path

import docx
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.shared import Pt


def set_landscape(doc: docx.document.Document) -> None:
    for sec in doc.sections:
        if sec.page_width < sec.page_height:
            sec.page_width, sec.page_height = sec.page_height, sec.page_width
        sec.orientation = WD_ORIENT.LANDSCAPE
        sec.left_margin = sec.right_margin = Pt(36)   # 0.5 英寸，给表格留出宽度


def _usable_width(sec) -> int:
    return sec.page_width - sec.left_margin - sec.right_margin


def fit_tables(doc: docx.document.Document, weights_by_ncols: dict | None = None) -> int:
    """按可用页宽重新分配列宽；行不允许跨页断开；表头每页重复。"""
    weights_by_ncols = weights_by_ncols or {}
    usable = _usable_width(doc.sections[0])
    n = 0
    for tb in doc.tables:
        ncols = len(tb.columns)
        tb.alignment = WD_TABLE_ALIGNMENT.CENTER
        tb.autofit = False
        # 固定表格布局，否则 Word 会按内容撑宽而溢出页面
        tblPr = tb._tbl.tblPr
        layout = tblPr.find(qn("w:tblLayout"))
        if layout is None:
            layout = tblPr.makeelement(qn("w:tblLayout"), {})
            tblPr.append(layout)
        layout.set(qn("w:type"), "fixed")

        # 表格总宽必须显式设为可用页宽，否则 Word/LibreOffice 只按单元格宽度
        # 排版，表格会缩在页面中间、列被挤窄。
        tblW = tblPr.find(qn("w:tblW"))
        if tblW is None:
            tblW = tblPr.makeelement(qn("w:tblW"), {})
            tblPr.append(tblW)
        tblW.set(qn("w:type"), "dxa")
        tblW.set(qn("w:w"), str(int(usable / 635)))      # EMU → dxa(1/20 pt)
        ind = tblPr.find(qn("w:tblInd"))
        if ind is not None:
            ind.set(qn("w:w"), "0")

        w = weights_by_ncols.get(ncols, [1.0] * ncols)
        total = float(sum(w))
        widths = [int(usable * x / total) for x in w]
        for row in tb.rows:
            # 行不跨页断开
            trPr = row._tr.get_or_add_trPr()
            if trPr.find(qn("w:cantSplit")) is None:
                trPr.append(trPr.makeelement(qn("w:cantSplit"), {}))
            for cell, width in zip(row.cells, widths):
                cell.width = width
        # 表头行每页重复
        if tb.rows:
            trPr = tb.rows[0]._tr.get_or_add_trPr()
            if trPr.find(qn("w:tblHeader")) is None:
                trPr.append(trPr.makeelement(qn("w:tblHeader"), {}))
        n += 1
    return n


if __name__ == "__main__":
    path = Path(sys.argv[1])
    landscape = "--landscape" in sys.argv
    doc = docx.Document(str(path))
    if landscape:
        set_landscape(doc)
    # 五列对照表：原稿/修订后/理由 需要更多宽度，步骤与性质列窄
    count = fit_tables(doc, weights_by_ncols={5: [0.7, 2.0, 2.4, 3.4, 1.0],
                                              4: [1.0, 2.6, 2.6, 1.2],
                                              3: [1.4, 2.6, 1.0],
                                              2: [1.0, 2.4]})
    doc.save(str(path))
    print(f"{path.name}: {count} 个表格已调整"
          f"{'（整篇改为横向）' if landscape else ''}")
