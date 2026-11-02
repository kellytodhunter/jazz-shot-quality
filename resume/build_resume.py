"""Build the tailored one-page resume PDF (reportlab)."""
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable

LI = "https://www.linkedin.com/in/kelly-todhunter-906b8830b/"
GH = "https://github.com/kellytodhunter"
REPO = GH + "/jazz-shot-quality"
KBO = GH + "/KBO-Player-Analysis"
NHL = GH + "/NHL-Comeback-Analysis"

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
D = "/System/Library/Fonts/Supplemental/Times New Roman"
for n, suf in [("TNR", ""), ("TNR-B", " Bold"), ("TNR-I", " Italic"), ("TNR-BI", " Bold Italic")]:
    pdfmetrics.registerFont(TTFont(n, f"{D}{suf}.ttf"))
registerFontFamily("TNR", normal="TNR", bold="TNR-B", italic="TNR-I", boldItalic="TNR-BI")
F, FB, FI = "TNR", "TNR-B", "TNR-I"
SZ = 10
name = ParagraphStyle("n", fontName=FB, fontSize=18, leading=21, alignment=TA_CENTER)
contact = ParagraphStyle("c", fontName=F, fontSize=SZ, leading=12, alignment=TA_CENTER)
sec = ParagraphStyle("s", fontName=FB, fontSize=10.4, leading=12, spaceBefore=5, spaceAfter=1)
txt = ParagraphStyle("t", fontName=F, fontSize=SZ, leading=11.8)
bul = ParagraphStyle("b", parent=txt, leftIndent=13, bulletIndent=3, spaceAfter=0.6)
right = ParagraphStyle("r", parent=txt, alignment=2)

def link(t, url): return f'<a href="{url}" color="#1a3c8c"><u>{t}</u></a>'

def head(left, date):
    t = Table([[Paragraph(left, txt), Paragraph(date, right)]], colWidths=[4.85 * inch, 2.1 * inch])
    t.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                           ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 0.5),
                           ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    return t

S = []
def section(title):
    S.append(Paragraph(title, sec)); S.append(HRFlowable(width="100%", thickness=0.7, color="black", spaceAfter=1.5))
def b(t): S.append(Paragraph(t, bul, bulletText="•"))

S.append(Paragraph("Kelly Todhunter", name))
S.append(Paragraph(f"Lafayette, CA • (848) 444-3867 • kellytodhunter@gmail.com • {link('LinkedIn', LI)} • {link('GitHub', GH)}", contact))

section("EDUCATION")
S.append(head("<b>Chapman University</b>, Orange, CA", "Expected Graduation May 2027"))
S.append(head("Bachelor of Science in Computer Science | Minor in Data Analytics", "4.00 GPA"))
S.append(Paragraph("Relevant Coursework: Data Science, Business Analytics", txt))

section("TECHNICAL SKILLS")
S.append(Paragraph("<b>Modeling &amp; analysis:</b> Python (pandas, NumPy, SciPy, scikit-learn), empirical Bayes, calibration and backtesting &nbsp;|&nbsp; "
                   "<b>Data:</b> SQL (DuckDB), web scraping (requests, BeautifulSoup) &nbsp;|&nbsp; "
                   "<b>Visualization:</b> matplotlib, Tableau, Advanced Excel &nbsp;|&nbsp; "
                   "<b>Engineering:</b> Git, pytest, Java, C++, JavaScript", txt))

section("SPORTS ANALYTICS PROJECTS")
S.append(head(f"<b>Shot Quality &amp; Shot-Making Model: Utah Jazz Analysis</b> | Python, scikit-learn, DuckDB, pytest | {link('GitHub', REPO)}", "November 2026"))
b("Built an expected-points-per-shot model on 870K NBA shots (2021-25); on a held-out 2024-25 season it beat a zone-average baseline on log loss by 0.022 (95% CI 0.021-0.023) and was well calibrated")
b("Estimated player shot-making with empirical-Bayes shrinkage and credible intervals; shrunk estimates cut next-season forecast error ~8% vs. raw estimates across 247 players")
b("Wrote a one-page coach-facing brief on the Jazz: low rim rate (27th of 30 in shot quality) and opponents shooting +4.6 pts/100 above expectation (29th), with uncertainty and limits stated")
b("Shipped a reproducible pipeline: validated ingestion, SQL aggregates, 26 unit tests, saved models, scoring CLI")
S.append(head(f"<b>KBO Marcel Projection System</b> | Python, pandas, NumPy, SciPy, matplotlib | {link('GitHub', KBO)}", "May 2026 - July 2026"))
b("Built the first Marcel-style projection system calibrated entirely on Korean Baseball Organization data, projecting 160 qualified hitters across five statistics")
b("Applied empirical Bayes regression to the mean with a cross-validated shrinkage parameter and delta-method aging curves to correct for survivorship bias")
b("Backtested on four held-out seasons (2021-2024), outperforming prior-year and league-average baselines; Spearman &rho; = 0.688 (p = 0.007) against mid-season 2026 standings")
S.append(head(f"<b>NHL 0-3 Playoff Comeback Analysis</b> | Python, pandas, matplotlib | {link('GitHub', NHL)}", "July 2026 - September 2026"))
b("Scraped game-by-game data for all 7 historical 0-3 comebacks and compared teams across 6 variables to find 3 shared conditions, noting the limits of a small sample")
b("Produced 12 visualizations and an interactive slideshow to explain findings to non-hockey audiences")

section("PROFESSIONAL EXPERIENCE")
S.append(head("<b>Research Intern</b> | Human-Inspired AI Research Center, Korea University - Seoul, Korea", "May 2026 - July 2026"))
b("Independently designed and executed a sports analytics research project, from research question to a manuscript submitted to the Asian American Interdisciplinary Research Journal (AAIRJ); published code and data pipeline on GitHub")
b("Collaborated with fellow interns to compile and review research literature for the center's annual conference")
S.append(head("<b>YouTube Analytics Consultant</b> | Independent Content Creator", "September 2026 - Present"))
b("Analyzed 45 videos across retention, CTR, traffic sources and subscriber conversion; built a Tableau KPI dashboard and presented data-driven recommendations to a non-technical client")
S.append(head("<b>Lacrosse Coach</b> | Frenzy Lacrosse Club - Orange, CA", "September 2025 - Present"))
b("Break down game film to identify tactical trends, turn them into practice plans, and explain adjustments to players and parents in clear terms")
S.append(head("<b>Data Worker</b> | The Salvation Army - Lafayette, CA", "September 2021 - June 2023"))
b("Collected and analyzed program-outcome data, ensuring accurate and on-time reporting")

section("ACADEMIC AWARDS AND LEADERSHIP")
S.append(head("NCAA Division III Student-Athlete, Chapman Women's Lacrosse", "August 2024 - October 2026"))
S.append(head("Provost's List", "2025, 2026"))
S.append(head("Presidential Scholarship; Fowler School Talent Scholarship", "2024, 2025, 2026"))

out = sys.argv[1]
SimpleDocTemplate(out, pagesize=letter, leftMargin=0.7 * inch, rightMargin=0.7 * inch, topMargin=0.45 * inch,
                  bottomMargin=0.4 * inch, title="Kelly Todhunter - Resume", author="Kelly Todhunter").build(S)
