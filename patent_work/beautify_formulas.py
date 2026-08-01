"""把说明书中的公式由"程序写法"排成正式数学写法（2026-07-31 反馈：下划线太难看）。

做两件事：
1. 希腊字母名与运算符还原为符号：lambda→λ、tau→τ、alpha→α、sqrt(...)→√(...)、
   ||x||→‖x‖、<=→≤、^2 的 2 变真上标等；
2. 把 `X_{abc}` / `X_abc` / `X^abc` 转成 Word 的**真下标／真上标运行格式**——
   这样 D_obs,i 会显示为 D 带小号下标 obs,i，而不是一个可见的下划线。
   之所以不插入 OMML 公式对象：既有公式(1)–(25)已经过多轮审阅，逐条重排为公式
   对象需要解析器，有引入转写错误的风险；运行格式在视觉上已达到正式排版要求。

安全性：转换后会把带格式的运行**逆向还原**为纯文本，与转换前逐字比对，
不一致即报错，确保只改排版不改内容。

用法：python3 beautify_formulas.py <docx>
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import docx

GREEK = {
    "lambda": "λ", "alpha": "α", "beta": "β", "gamma": "γ", "theta": "θ",
    "sigma": "σ", "Sigma": "Σ", "tau": "τ", "rho": "ρ", "eta": "η",
    "phi": "φ", "psi": "ψ", "mu": "μ", "epsilon": "ε", "delta": "δ",
}
# 下标／上标只允许由这些字符构成：拉丁字母、数字、逗号（如 D_obs,i）、
# 下划线（如 D_mat_missing,i）、正负号（如 10^-6）。遇到中文、空格、运算符即终止，
# 避免把"u_t为未归一化法向"整句吞进下标。
_BASE = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789,+-")
SUB_CHARS = _BASE | {"_"}      # 下标可含下划线，如 D_mat_missing,i
SUP_CHARS = _BASE              # 上标不含下划线，否则 Q^res_i 会把 _i 吞进上标


_FORMULA_NO = re.compile(r"（\s*\d+(?:\s*-\s*\d+)?\s*）\s*$")


def _symbols(s: str) -> str:
    """希腊字母名与运算符符号化（不触碰上下标结构）。"""
    # 段尾的公式编号（如"（9-1）"）先摘出，避免其中的连字符被当成减号替换
    tail = ""
    m = _FORMULA_NO.search(s)
    if m:
        tail, s = s[m.start():], s[:m.start()]
    for name, ch in sorted(GREEK.items(), key=lambda kv: -len(kv[0])):
        s = re.sub(rf"(?<![A-Za-z]){name}(?![A-Za-z])", ch, s)
    s = re.sub(r"\|\|([^|]+)\|\|", r"‖\1‖", s)      # ||x-y|| → ‖x−y‖
    s = re.sub(r"sqrt\s*[\[(]([^\])]+)[\])]", r"√(\1)", s)
    s = s.replace("<=", "≤").replace(">=", "≥")
    # 减号：仅在两侧均为数学符号时替换；重复应用以覆盖 a-b-c 这类连续情形
    for _ in range(4):
        s2 = re.sub(r"(?<=[\w)\]}])\s*-\s*(?=[\w(\[{])", " − ", s)
        if s2 == s:
            break
        s = s2
    return s + tail


def _split_runs(s: str) -> list[tuple[str, str]]:
    """切成 (文本, 样式) 序列，样式 ∈ {'', 'sub', 'sup'}。"""
    out: list[tuple[str, str]] = []
    buf = ""
    i = 0
    while i < len(s):
        c = s[i]
        if c in "_^" and i + 1 < len(s):
            style = "sub" if c == "_" else "sup"
            j = i + 1
            if s[j] == "{":
                k = s.find("}", j)
                if k > 0:
                    if buf:
                        out.append((buf, "")); buf = ""
                    out.append((s[j + 1:k], style))
                    i = k + 1
                    continue
            elif s[j] == "(":
                k = s.find(")", j)
                if k > 0:
                    if buf:
                        out.append((buf, "")); buf = ""
                    out.append((s[j + 1:k], style))
                    i = k + 1
                    continue
            allowed = SUB_CHARS if style == "sub" else SUP_CHARS
            k = j
            while k < len(s) and s[k] in allowed:
                # 逗号后接数字的，是函数实参而非下标的一部分：
                # D_obs,i 的下标是"obs,i"，而 clip(ρ_i,0,1) 的下标只是"i"。
                if s[k] == "," and (k + 1 >= len(s) or s[k + 1].isdigit()):
                    break
                k += 1
            token = s[j:k].rstrip(",")          # 末尾逗号属于正文而非下标
            # "material_conflict_i" 这类：名字本身由下划线连接，真正的下标只是
            # 末尾那个单字符。若不区分，会错排成 material + 下标 conflict_i。
            m2 = re.fullmatch(r"([a-z]+)_([0-9a-z])", token)
            if style == "sub" and m2:
                buf += "_" + m2.group(1)
                out.append((buf, "")); buf = ""
                out.append((m2.group(2), style))
                i = j + len(token)
                continue
            if token:
                if buf:
                    out.append((buf, "")); buf = ""
                out.append((token, style))
                i = j + len(token)
                continue
        buf += c
        i += 1
    if buf:
        out.append((buf, ""))
    return out


_UNI_SUB = {**{c: ch for c, ch in zip("0123456789", "₀₁₂₃₄₅₆₇₈₉")},
            **{c: ch for c, ch in zip("aehijklmnoprstuvx", "ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ")}}


def _nest(token: str) -> str:
    """把下标内部的嵌套单字符下标转为 Unicode 下标：T_i → Tᵢ。

    仅当下划线后仅有一个字符、且其后不再接字母数字时才转换，
    以免把 mat_missing 这类由下划线连接的名字拆错。
    """
    return re.sub(r"_([0-9a-z])(?![A-Za-z0-9])",
                  lambda m: _UNI_SUB.get(m.group(1), "_" + m.group(1)), token)


def _restore(parts: list[tuple[str, str]]) -> str:
    """逆向还原为纯文本，用于校验转换未改动内容。"""
    out = ""
    for text, style in parts:
        if style == "sub":
            out += "_" + (("{" + text + "}") if len(text) > 1 else text)
        elif style == "sup":
            out += "^" + (("{" + text + "}") if len(text) > 1 else text)
        else:
            out += text
    return out


def _norm(s: str) -> str:
    """去掉上下标外围的花括号／圆括号，使"原文"与"还原结果"可逐字比对。

    原文中 p_{t,0} 与 p_t,0 表示同一含义，还原时统一补花括号，故比对前一并去掉。
    """
    s = re.sub(r"_\{([^{}]*)\}", r"_\1", s)
    s = re.sub(r"\^\{([^{}]*)\}", r"^\1", s)
    s = re.sub(r"\^\(([^()]*)\)", r"^\1", s)
    return s


def beautify(paragraph) -> bool:
    text = paragraph.text
    if not re.search(r"[_^]\s*[{(\w]", text) and not any(
            re.search(rf"(?<![A-Za-z]){g}(?![A-Za-z])", text) for g in GREEK):
        return False
    converted = _symbols(text)
    parts = _split_runs(converted)
    if not any(st for _, st in parts):
        if converted == text:
            return False
    # 校验：还原后应与符号化文本一致（只改排版不改内容）
    assert _norm(_restore(parts)) == _norm(converted), f"转换不可逆:\n{text}"

    style = paragraph.runs[0].style if paragraph.runs else None
    bold = paragraph.runs[0].bold if paragraph.runs else None
    for r in list(paragraph.runs):
        r._element.getparent().remove(r._element)
    for txt, st in parts:
        run = paragraph.add_run(_nest(txt) if st == "sub" else txt)
        if style is not None:
            run.style = style
        if bold:
            run.bold = True
        if st == "sub":
            run.font.subscript = True
        elif st == "sup":
            run.font.superscript = True
    return True


if __name__ == "__main__":
    path = Path(sys.argv[1])
    doc = docx.Document(str(path))
    # 只处理说明书正文（权利要求书中无上下标记号），跳过标题
    n = 0
    for p in doc.paragraphs:
        if re.match(r"^\d+\.\s*一种", p.text.strip()):     # 权利要求段落
            continue
        if beautify(p):
            n += 1
    doc.save(str(path))
    print(f"{path.name}: {n} 个段落已排版为真上下标")
