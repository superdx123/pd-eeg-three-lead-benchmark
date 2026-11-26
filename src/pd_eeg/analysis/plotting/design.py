"""Two-cohort data preparation and evaluation design."""

from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.patches import Arc, Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

ROLE_COLORS = {
    "train": {"fill": "#B5D1E7", "edge": "#35658B"},
    "validation": {"fill": "#F3E2AF", "edge": "#8D6C26"},
    "test": {"fill": "#F3BAB2", "edge": "#9B4C43", "pale": "#FBE3DE"},
}
NAVY, MUTED, OUTLINE = "#26313A", "#515B62", "#A8AFB4"
WHITE, HEADER, NEUTRAL, MID_NEUTRAL = "#FFFFFF", "#E9EBED", "#F4F5F6", "#E0E3E5"
PREP_FILL, PREP_EDGE = "#EAE5F2", "#877E96"
AGG_FILL, AGG_EDGE = "#E7F0EC", "#8AA79B"


def draw(output, formats=("pdf", "svg", "png"), dpi=600):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "svg.fonttype": "none",
                         "svg.hashsalt": "bspc-fig1-v051", "axes.unicode_minus": False})
    WIDTH, HEIGHT = 1000, 712
    fig = plt.figure(figsize=(7.1, 7.1 * HEIGHT / WIDTH), dpi=160, facecolor=WHITE)
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, WIDTH), ylim=(HEIGHT, 0))
    ax.set_axis_off()
    labels = []
    role_artists = {}


    def label(x, y, value, size=8.8, bold=False, color=NAVY, align="center"):
        t = ax.text(x, y, value, fontsize=size, fontweight="bold" if bold else "normal",
                    color=color, ha=align, va="center", zorder=5, linespacing=1.2)
        labels.append(t)
        return t


    def box(x, y, w, h, fill=WHITE, edge=OUTLINE, radius=7, lw=.85, z=1):
        patch = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                               facecolor=fill, edgecolor=edge, lw=lw, zorder=z)
        ax.add_patch(patch)
        return patch


    def line(x1, y1, x2, y2, color=OUTLINE, lw=.7, z=3):
        return ax.plot((x1, x2), (y1, y2), color=color, lw=lw, zorder=z, solid_capstyle="round")[0]


    def arrow(x1, y1, x2, y2, color=NAVY):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=8,
                                    color=color, lw=.85, shrinkA=0, shrinkB=0, zorder=4))


    def panel(x, y, w, h, title):
        p = box(x, y, w, h)
        band = Rectangle((x, y), w, 36, facecolor=HEADER, edgecolor="none", zorder=2)
        band.set_clip_path(p)
        ax.add_patch(band)
        box(x, y, w, h, "none", OUTLINE, z=3)
        label(x + 12, y + 18, title, 10.5, True, align="left")


    def person(x, y, fill, edge=NAVY, scale=1):
        head = Circle((x, y - 7 * scale), 4.2 * scale, facecolor=fill,
                      edgecolor=edge, lw=.75, zorder=4)
        ax.add_patch(head)
        body = box(x - 7 * scale, y - 1 * scale, 14 * scale, 10 * scale,
                   fill, edge, radius=4 * scale, lw=.75, z=4)
        return head, body


    def tag_role(artist, role, name):
        assert role in ROLE_COLORS and name not in role_artists
        artist.set_gid(f"role-{role}-{name}")
        role_artists[name] = (role, artist)
        return artist


    def role_box(role, name, x, y, w, h, *, pale=False, **kwargs):
        colors = ROLE_COLORS[role]
        return tag_role(box(x, y, w, h, colors["pale" if pale else "fill"],
                            colors["edge"], **kwargs), role, name)


    def eye(x, y):
        ax.add_patch(Arc((x, y + 3), 24, 19, theta1=208, theta2=332, color=NAVY, lw=1, zorder=3))
        ax.add_patch(Arc((x, y - 3), 24, 19, theta1=28, theta2=152, color=NAVY, lw=1, zorder=3))
        ax.add_patch(Circle((x, y), 3.1, facecolor=NAVY, edgecolor="none", zorder=4))


    def cohort(x, dataset, total, hc, pd, recording):
        panel(x, 8, 486, 111, f"{dataset}  (n = {total})")
        person(x + 29, 81, WHITE, MUTED, 1.22)
        person(x + 150, 81, MUTED, MUTED, 1.22)
        label(x + 49, 69, "HC", 9.2, True, align="left")
        label(x + 49, 92, f"n = {hc}", 8.9, align="left")
        label(x + 165, 69, "PD (OFF)", 8.9, True, align="left")
        label(x + 165, 92, f"n = {pd}", 8.9, align="left")
        line(x + 119, 57, x + 119, 103)
        line(x + 248, 57, x + 248, 103)
        eye(x + 274, 78)
        label(x + 297, 69, "Eyes-open rest", 9.2, True, align="left")
        label(x + 297, 92, recording, 8.8, align="left")


    cohort(8, "ds002778", 31, 16, 15, "~3 min, 512 Hz")
    cohort(506, "ds003490", 50, 25, 25, "EO segments, 500 Hz")

    box(8, 137, 984, 111)
    box(16, 145, 152, 95, PREP_FILL, edge="none")
    label(92, 193, "Data\npreparation", 10.5, True)
    label(187, 165, "ds002778", 9.0, True, align="left")
    label(322, 165, "Prepared 7-channel windows: select C3 / Cz / C4", 8.45, align="left")
    line(184, 193, 681, 193)
    label(187, 219, "ds003490", 9.0, True, align="left")
    label(322, 219, "Extract EO; 3-channel QC and z-scoring", 8.45, align="left")
    arrow(698, 193, 728, 193)
    box(741, 150, 240, 85, PREP_FILL, PREP_EDGE, radius=5)
    label(861, 171, "Common classifier input", 9.5, True)
    label(861, 193, "C3 / Cz / C4   |   3 × 512", 9.1)
    label(861, 216, "2 s windows at 256 Hz", 8.6, color=MUTED)

    for x, title in ((8, "(a) S: Unseen subjects"),
                     (340, "(b) W: Within-recording"),
                     (672, "(c) Frozen transfer")):
        panel(x, 267, 320, 303, title)

    label(168, 325, "Outer LOSO: 31 or 50 folds", 9.1, True)
    box(20, 347, 184, 132, NEUTRAL, OUTLINE, radius=5)
    label(112, 366, "N − 1 subjects", 9.1, True)
    role_box("train", "s-train", 27, 384, 79, 34, radius=4)
    role_box("validation", "s-validation", 113, 384, 84, 34, radius=4)
    label(66.5, 401, "Train", 8.8, True)
    label(155, 401, "Validate", 8.8, True)
    label(112, 438, "5-fold inner subject CV", 8.45)
    label(112, 459, "when tuning is needed", 8.3, color=MUTED)
    role_box("test", "s-test", 218, 347, 98, 132, pale=True, radius=5)
    for part, shape in zip(("head", "body"), person(
            267, 372, ROLE_COLORS["test"]["fill"], ROLE_COLORS["test"]["edge"], 1.10)):
        tag_role(shape, "test", f"s-test-subject-{part}")
    label(267, 398, "Test", 9, True)
    label(267, 422, "1 subject", 8.8)
    label(267, 450, "All windows", 8.25)
    arrow(112, 481, 112, 500)
    box(23, 507, 288, 44, NEUTRAL, OUTLINE, radius=5)
    label(167, 529, "Final fit on all N − 1 subjects", 8.55, True)

    label(500, 325, "Five folds × five repeats", 9.1, True)
    label(500, 353, "Windows from the same subjects", 8.6, color=MUTED)
    for x, w, role, name, detail in (
            (391, 91, "train", "Train", "3 folds"),
            (492, 73, "validation", "Validate", "1 fold"),
            (575, 73, "test", "Test", "1 fold")):
        for row, y in enumerate((378, 414), start=1):
            role_box(role, f"w-{role}-{row}", x, y, w, 26, radius=4)
            for off in range(1, 3):
                split = line(x + w * off / 3, y + 5, x + w * off / 3, y + 21,
                             ROLE_COLORS[role]["edge"], .45)
                tag_role(split, role, f"w-{role}-{row}-divider-{off}")
        label(x + w / 2, 459, name, 8.7, True)
        label(x + w / 2, 479, detail, 8.5, color=MUTED)
    for y in (391, 427):
        person(365, y, WHITE, MUTED, .86)
        arrow(375, y, 385, y)
    arrow(499, 489, 499, 500)
    box(355, 507, 288, 44, NEUTRAL, OUTLINE, radius=5)
    label(499, 529, "Final fit on four non-test folds", 8.55, True)

    label(832, 325, "Both cohort directions", 9.1, True)
    for x, fill, edge, name in ((684, NEUTRAL, OUTLINE, "Source\ncohort"),
                                (789, MID_NEUTRAL, MUTED, "Frozen\nmodel"),
                                (894, NEUTRAL, OUTLINE, "Target\ncohort")):
        box(x, 350, 86, 73, fill, edge, radius=5)
        label(x + 43, 387, name, 8.9, True)
    arrow(773, 387, 786, 387)
    arrow(878, 387, 891, 387)
    label(832, 445, "Freeze weights and source scaling", 8.65)
    label(832, 468, "Score target cohort only", 8.65)
    box(686, 490, 292, 61, NEUTRAL, OUTLINE, radius=5)
    label(832, 503, "BE-MSNet / ATCNet", 8.0, True)
    label(832, 521, "Select; refit on full source", 8.0)
    label(832, 539, "Spectral LR: fixed C = 1", 8.0)

    for x in (168, 500, 832):
        arrow(x, 575, x, 591)
    box(8, 597, 984, 64, AGG_FILL, AGG_EDGE)
    label(152, 618, "Window probabilities", 9.55, True)
    arrow(279, 618, 306, 618)
    label(432, 618, "Mean within subject", 9.55, True)
    arrow(558, 618, 585, 618)
    label(766, 618, "Equal-weight mean across seeds", 9.55, True)
    label(500, 644, "Score at the subject level; for W, average the scores over 25 test folds.", 8.6,
          color=MUTED)

    for x, role, name in ((104, "train", "Training"), (265, "validation", "Validation"),
                           (438, "test", "Test")):
        role_box(role, f"legend-{role}", x, 681, 26, 14, radius=3, lw=.65)
        label(x + 36, 688, name, 8.45, align="left")
    line(556, 676, 556, 701)
    label(578, 688, "HC: healthy control   |   PD: Parkinson’s disease", 8.45, align="left")

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    extents = [t.get_window_extent(renderer).expanded(1.015, 1.015) for t in labels]
    for label_a, extent in zip(labels, extents):
        assert fig.bbox.contains(*extent.p0) and fig.bbox.contains(*extent.p1), label_a.get_text()
    for (ta, ea), (tb, eb) in combinations(zip(labels, extents), 2):
        assert not ea.overlaps(eb), (ta.get_text(), tb.get_text())

    selection = next(t for t in labels if t.get_text() == "BE-MSNet / ATCNet")
    selection_bounds = selection.get_window_extent(renderer).transformed(ax.transData.inverted())
    assert selection_bounds.x0 >= 691 and selection_bounds.x1 <= 973, selection_bounds


    def artist_colors(artist):
        if hasattr(artist, "get_facecolor"):
            return {to_hex(color).upper() for color in (
                artist.get_facecolor(), artist.get_edgecolor()) if color[-1] > 0}
        return {to_hex(artist.get_color()).upper()}


    reserved = {color for colors in ROLE_COLORS.values() for color in colors.values()}
    expected = {artist: (role, name) for name, (role, artist) in role_artists.items()}
    observed = set()
    for artist in [*ax.patches, *ax.lines, *ax.texts]:
        used = artist_colors(artist) & reserved
        if artist in expected:
            role, name = expected[artist]
            assert used and used <= set(ROLE_COLORS[role].values()), (name, used)
            observed.add(artist)
        else:
            assert not used, (artist, used)
    assert set(expected) == observed

    for ext in formats:
        fig.savefig(output / f"design_overview.{ext}", dpi=dpi, facecolor=WHITE)

    plt.close(fig)
    return [f"design_overview.{ext}" for ext in formats]
