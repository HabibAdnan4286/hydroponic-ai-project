"""Step 5a - PRISMA 2020-style flow diagram from YOUR real screening counts.

Fill paper/review/config_prisma.json with the counts from your screening log. The script:
  * refuses to draw while any count is still null (so no placeholder numbers can slip into the paper),
  * checks the arithmetic (identified - duplicates = screened, etc.) and stops on any mismatch.

Run:  python paper\\scripts\\make_prisma.py
"""
import json

from matplotlib.patches import FancyBboxPatch

from common import REVIEW, plt, savefig

cfg = json.load(open(REVIEW / "config_prisma.json", encoding="utf-8"))
ident, ft_ex = cfg["identification"], cfg["excluded_fulltext"]

nulls = [k for k, v in ident.items() if v is None] + [k for k, v in ft_ex.items() if v is None] + \
        [k for k in ("duplicates_removed", "excluded_title_abstract", "included") if cfg[k] is None]
if nulls:
    raise SystemExit("config_prisma.json still has empty counts: " + ", ".join(nulls) +
                     "\nFill them from your screening log (use 0 where a category has no records).")

total_ident = sum(ident.values())
screened = total_ident - cfg["duplicates_removed"]
fulltext = screened - cfg["excluded_title_abstract"]
ft_excluded = sum(ft_ex.values())
included_calc = fulltext - ft_excluded
if included_calc != cfg["included"]:
    raise SystemExit(f"Arithmetic mismatch: records after full-text exclusions = {included_calc}, but 'included' = {cfg['included']}.")
if min(screened, fulltext, included_calc) < 0:
    raise SystemExit("Negative count - check the numbers.")

fig, ax = plt.subplots(figsize=(7.4, 8.6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis("off")


def box(x, y, w, h, text, fc="#EAF2F8", bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.1", fc=fc, ec="#333333", lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8, fontweight="bold" if bold else "normal")


def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1))


db_text = f"Records identified (n = {total_ident})\n" + "\n".join(f"{k}: {v}" for k, v in ident.items())
box(0.3, 9.4, 4.6, 2.4, db_text)
box(0.3, 7.7, 4.6, 1.2, f"Records after duplicates removed\n(n = {screened})")
box(5.3, 7.7, 4.4, 1.2, f"Duplicates removed\n(n = {cfg['duplicates_removed']})", fc="#FDEDEC")
box(0.3, 5.9, 4.6, 1.2, f"Records screened (title / abstract)\n(n = {screened})")
box(5.3, 5.9, 4.4, 1.2, f"Excluded at title / abstract\n(n = {cfg['excluded_title_abstract']})", fc="#FDEDEC")
box(0.3, 4.1, 4.6, 1.2, f"Full-text articles assessed for eligibility\n(n = {fulltext})")
box(5.3, 2.6, 4.4, 2.7, f"Full-text articles excluded (n = {ft_excluded})\n" + "\n".join(f"{k}: {v}" for k, v in ft_ex.items()), fc="#FDEDEC")
box(0.3, 2.3, 4.6, 1.2, f"Studies included in the review\n(n = {cfg['included']})", fc="#E8F6EF", bold=True)
arrow(2.6, 9.4, 2.6, 8.9)
arrow(4.9, 8.3, 5.3, 8.3)
arrow(2.6, 7.7, 2.6, 7.1)
arrow(4.9, 6.5, 5.3, 6.5)
arrow(2.6, 5.9, 2.6, 5.3)
arrow(4.9, 4.7, 5.3, 4.7)
arrow(2.6, 4.1, 2.6, 3.5)
for y, label in [(10.6, "Identification"), (6.5, "Screening"), (2.9, "Included")]:
    ax.text(0.0, y, label, rotation=90, va="center", ha="center", fontsize=8, fontweight="bold", color="#555555")
savefig(fig, "fig_r1_prisma_flow")
print("PRISMA arithmetic checks passed.")
