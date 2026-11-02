"""Render reports/jazz_brief.md (headings, bullets, tables) plus the two Jazz figures to a PDF."""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import letter  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import inch  # noqa: E402
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle  # noqa: E402

from shotquality import config  # noqa: E402

NAVY = colors.HexColor("#002B5C")
ss = getSampleStyleSheet()
body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9.2, leading=12)
h1 = ParagraphStyle("h1", parent=ss["Title"], fontSize=17, textColor=NAVY, alignment=0, spaceAfter=4)
h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=11.5, textColor=NAVY, spaceBefore=8, spaceAfter=3)
cell = ParagraphStyle("c", parent=body, fontSize=8.4, leading=10.5)


def inline(t: str) -> str:
    t = t.replace("&", "&amp;").replace("<", "&lt;")
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    return re.sub(r"(?<!\*)\*(.+?)\*(?!\*)", r"<i>\1</i>", t)


def table(rows):
    data = [[Paragraph(inline(c), cell) for c in r] for r in rows]
    t = Table(data, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E6ECF4")),
                           ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B8C0CC")),
                           ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 2),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
    return t


def build(out: Path) -> None:
    lines = (config.REPORT_DIR / "jazz_brief.md").read_text().splitlines()
    story, rows = [], []

    def flush():
        nonlocal rows
        if rows:
            story.extend([table(rows), Spacer(1, 5)])
            rows = []

    for ln in lines:
        if ln.startswith("|"):
            cells = [c.strip() for c in ln.strip("|").split("|")]
            if not all(re.fullmatch(r"-+", c) for c in cells):
                rows.append(cells)
            continue
        flush()
        if ln.startswith("# "):
            story.append(Paragraph(inline(ln[2:]), h1))
        elif ln.startswith("## "):
            story.append(Paragraph(inline(ln[3:]), h2))
            if ln.startswith("## Shot-making by zone"):
                pass
        elif re.match(r"\d+\. ", ln):
            story.append(Paragraph(inline(ln), body))
        elif ln.strip():
            story.append(Paragraph(inline(ln), body))
    flush()
    story.append(Spacer(1, 6))
    story.append(Image(str(config.FIG_DIR / "fig4_jazz_zones.png"), width=6.6 * inch, height=2.5 * inch))
    story.append(Image(str(config.FIG_DIR / "fig3_jazz_shotmaking.png"), width=4.6 * inch, height=3.2 * inch))
    SimpleDocTemplate(str(out), pagesize=letter, leftMargin=0.7 * inch, rightMargin=0.7 * inch,
                      topMargin=0.6 * inch, bottomMargin=0.6 * inch, title="Jazz 2024-25 shot quality brief").build(story)


if __name__ == "__main__":
    out = config.REPORT_DIR / "jazz_brief.pdf"
    build(out)
    print("wrote", out)
