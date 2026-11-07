#!/usr/bin/env python3
"""Embed generated PNG figures into a DOCX report at marker paragraphs.

The editor SDK's Markdown importer turns `![](path)` into literal text, so the
report is assembled in two steps: (1) insert Markdown containing standalone
`@@FIGn@@` marker paragraphs; (2) replace each marker paragraph with an OOXML
inline drawing and write the PNG bytes into word/media/.

Pure standard library; no Word automation required.  Kept in the repository so
the report itself is reproducible.

Usage:
    python src/embed_figures.py --in report_nofigs.docx --out report.docx \
        --map "@@FIG1@@=reports/figures/fig1_monthly_volume.png" ...
"""
from __future__ import annotations

import argparse
import re
import struct
import zipfile
from pathlib import Path

NS = (
    'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
    'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
    'xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture" '
    'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"'
)
REL_TYPE = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"
TARGET_WIDTH_EMU = 5_600_000  # ~6.1 inch, fits a 2 cm-margin A4/Letter page


def png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as f:
        head = f.read(24)
    if head[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"not a PNG: {path}")
    return struct.unpack(">II", head[16:24])


def image_paragraph(rid: str, idx: int, w: int, h: int) -> str:
    cx = TARGET_WIDTH_EMU
    cy = int(TARGET_WIDTH_EMU * h / w)
    return (
        '<w:p><w:pPr><w:jc w:val="center"/></w:pPr><w:r><w:drawing ' + NS + '>'
        f'<wp:inline distT="0" distB="0" distL="0" distR="0">'
        f'<wp:extent cx="{cx}" cy="{cy}"/>'
        f'<wp:effectExtent l="0" t="0" r="0" b="0"/>'
        f'<wp:docPr id="{1000 + idx}" name="Phase2 Figure {idx}"/>'
        f'<wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/>'
        f'</wp:cNvGraphicFramePr>'
        f'<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        f'<pic:pic><pic:nvPicPr><pic:cNvPr id="0" name="fig{idx}.png"/>'
        f'<pic:cNvPicPr/></pic:nvPicPr>'
        f'<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
        f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
        f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>'
        f'</pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>'
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--map", nargs="+", required=True,
                    help='marker=path pairs, e.g. "@@FIG1@@=reports/figures/fig1.png"')
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    pairs = []
    for item in args.map:
        marker, path = item.split("=", 1)
        pairs.append((marker, Path(path)))

    with zipfile.ZipFile(src) as z:
        items = {n: z.read(n) for n in z.namelist()}
    doc = items["word/document.xml"].decode("utf-8")
    rels = items.get("word/_rels/document.xml.rels", b"<Relationships/>").decode("utf-8")
    ct = items["[Content_Types].xml"].decode("utf-8")

    if 'Extension="png"' not in ct:
        ct = ct.replace("</Types>",
                        '<Default Extension="png" ContentType="image/png"/></Types>')
    used = [int(x) for x in re.findall(r'Id="rId(\d+)"', rels)]
    next_id = max(used or [0]) + 1

    para_re = re.compile(r"<w:p(?:\s[^>]*)?>.*?</w:p>", re.S)
    new_rels, embedded = [], []
    for i, (marker, png) in enumerate(pairs, 1):
        found = False

        def repl(m):
            nonlocal found
            if found or marker not in m.group(0):
                return m.group(0)
            found = True
            return image_paragraph(f"rId{next_id + i - 1}", i, *png_size(png))

        doc = para_re.sub(repl, doc)
        if not found:
            raise ValueError(f"marker not found in document: {marker}")
        media_name = f"word/media/phase2_fig{i}.png"
        items[media_name] = png.read_bytes()
        new_rels.append(
            f'<Relationship Id="rId{next_id + i - 1}" Type="{REL_TYPE}" '
            f'Target="media/phase2_fig{i}.png"/>')
        embedded.append((marker, str(png), png_size(png)))

    rels = rels.replace("</Relationships>", "".join(new_rels) + "</Relationships>")
    items["word/document.xml"] = doc.encode("utf-8")
    items["word/_rels/document.xml.rels"] = rels.encode("utf-8")
    items["[Content_Types].xml"] = ct.encode("utf-8")

    dst.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in items.items():
            z.writestr(name, data)
    for marker, path, size in embedded:
        print(f"[embed] {marker} <- {path} ({size[0]}x{size[1]})")
    print(f"[embed] wrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
