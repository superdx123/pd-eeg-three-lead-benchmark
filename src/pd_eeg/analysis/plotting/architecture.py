"""BE-MSNet architecture and frequency-band construction."""

import hashlib
import json
import math
import os
from itertools import combinations
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt, font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.transforms import Bbox

P = json.loads((Path(__file__).resolve().parents[1] / "data/style.json").read_text(encoding="utf-8"))["palette"]
NAVY = P["flow_blue_outline"]
CHROME = "#F0F2F4"
ROLES = {
    "Input": ("#D7EAE3", "#548779"),
    "Spectral": ("#E4DCF1", "#8273A2"),
    "Temporal": ("#D2E3F1", "#658AAC"),
    "Features": ("#F2E5C6", "#A48345"),
    "Classifier": ("#F6D9D4", "#A76C63"),
}
INPUT, GREEN = ROLES["Input"]
SPECTRAL, PURPLE = ROLES["Spectral"]
TEMPORAL, BLUE = ROLES["Temporal"]
FEATURES, AMBER = ROLES["Features"]
CLASSIFIER, CORAL = ROLES["Classifier"]
EDGE, PAPER, MUTED = "#A1B3C1", "#FFFFFF", "#485766"
W, H = 7.10, 5.16
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "text.color": NAVY, "pdf.fonttype": 42, "svg.fonttype": "none",
                     "svg.hashsalt": "bspc-model-v051", "figure.facecolor": PAPER})


class Diagram:
    def __init__(self):
        self.fig = plt.figure(figsize=(W, H), dpi=180)
        self.ax = self.fig.add_axes((0, 0, 1, 1), xlim=(0, W), ylim=(H, 0))
        self.ax.set_axis_off()
        self.labels = []

    def text(self, x, y, label, size=9, *, bold=False, align="center", color=NAVY,
             box=None, rotation=0):
        artist = self.ax.text(x, y, label, fontsize=size, weight="bold" if bold else "normal",
                              color=color, ha=align, va="center", linespacing=1.1,
                              rotation=rotation, zorder=7)
        self.labels.append((artist, box))

    def rect(self, x, y, w, h, fill, edge=EDGE, lw=.70, r=.035, zorder=2):
        patch = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                               fc=fill, ec=edge, lw=lw, zorder=zorder)
        self.ax.add_patch(patch)
        return patch

    def panel(self, x, y, w, h, title):
        patch = self.rect(x, y, w, h, PAPER, zorder=0)
        ribbon = Rectangle((x, y), w, .264, fc=CHROME, ec="none", zorder=1)
        ribbon.set_clip_path(patch)
        self.ax.add_patch(ribbon)
        self.line((x, y+.264), (x+w, y+.264), color=EDGE, lw=.66)
        self.text(x+.105, y+.135, title, 11, bold=True, align="left")

    def node(self, x, cy, w, h, label, fill=CHROME, *, size=9, bold=False, edge=EDGE,
             rotation=0):
        y = cy-h/2
        self.rect(x, y, w, h, fill, edge)
        self.text(x+w/2, cy, label, size, bold=bold, box=(x, y, x+w, y+h),
                  rotation=rotation)

    def line(self, *pts, arrow=False, color=NAVY, lw=.76):
        begin, end = pts[-2:]
        track = pts[:-1] if arrow else pts
        if len(track) >= 2:
            self.ax.plot(*zip(*track), color=color, lw=lw, solid_capstyle="round",
                         solid_joinstyle="round", zorder=3)
        if arrow:
            self.ax.add_patch(FancyArrowPatch(begin, end, arrowstyle="-|>", mutation_scale=7.4,
                                             shrinkA=0, shrinkB=0, color=color, lw=lw, zorder=4))

    def grid(self, x, cy, w=.29, h=.181, nx=4, ny=3, stacked=True, fill=TEMPORAL, edge=BLUE):
        y = cy-h/2
        if stacked:
            for shift in (.048, .024):
                self.ax.add_patch(Rectangle((x+shift, y-shift), w, h, lw=.57,
                                            fc=PAPER, ec=edge, zorder=4))
        self.ax.add_patch(Rectangle((x, y), w, h, lw=.61, fc=fill, ec=edge, zorder=5))
        for i in range(1, nx):
            self.ax.plot([x+i*w/nx]*2, [y,y+h], color=edge, lw=.42, zorder=6)
        for j in range(1, ny):
            self.ax.plot([x,x+w], [y+j*h/ny]*2, color=edge, lw=.42, zorder=6)

    def wave(self, x, cy, w, n, phase, amp=.022, color="#32695D"):
        tx = [j/129 for j in range(130)]
        ty = [cy + amp*(math.sin(2*math.pi*n*t+phase)
                       + .22*math.sin(2*math.pi*(n+1.8)*t+.3)) for t in tx]
        self.ax.plot([x+w*t for t in tx], ty, color=color, lw=.75, zorder=5)

    def verify(self):
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        inv = self.ax.transData.inverted()
        labels = []
        for artist, node in self.labels:
            raw = artist.get_window_extent(renderer).transformed(inv)
            b = Bbox.from_extents(min(raw.x0,raw.x1), min(raw.y0,raw.y1),
                                  max(raw.x0,raw.x1), max(raw.y0,raw.y1))
            assert .034 < b.x0 < b.x1 < W-.034 and .034 < b.y0 < b.y1 < H-.034, artist.get_text()
            if node:
                l, top, r, bottom = node
                assert l+.015 <= b.x0 and b.x1 <= r-.015 and top+.009 <= b.y0 and b.y1 <= bottom-.009, (artist.get_text(), list(b.extents), node)
            labels.append((artist.get_text(), b))
        for (a,p),(b,q) in combinations(labels,2):
            assert min(p.x1,q.x1)-max(p.x0,q.x0) <= .008 or min(p.y1,q.y1)-max(p.y0,q.y0) <= .008, ("text overlap", a,b)
        return len(labels)


def build():
    d = Diagram()
    d.panel(.055, .052, 6.99, 1.97, "(a)  BE-MSNet architecture")
    for x, (role, (fill, edge)) in zip((2.500,3.205,4.086,5.023,5.906), ROLES.items()):
        d.rect(x, .126, .126, .116, fill, edge, lw=.67, r=.018)
        d.text(x+.184, .183, role, 8.6, align="left")
    d.rect(.160, .703, .88, 1.001, INPUT, GREEN)
    d.text(.600, .845, "Input", 9.5, bold=True)
    for i, channel in enumerate(("C3", "Cz", "C4")):
        y = 1.027+i*.169
        d.text(.233, y, channel, 8.55, align="left")
        d.wave(.441, y, .499, 3.1+i*.45, i*.7)
    d.text(.600, 1.565, "3 × 512", 9.1, color=MUTED)
    d.line((1.055,1.190),(1.215,1.190),arrow=True)
    d.node(1.235,1.190,.819,.66,"Band\nexpansion\n18 × 512",SPECTRAL,bold=True,edge=PURPLE)
    d.text(1.645,1.639,"see (b)",8.5,color=MUTED)
    d.line((2.066,1.190),(2.170,1.190))
    d.line((2.170,.816),(2.170,1.564))
    d.text(2.742,.478,"Temporal branches",9.2,bold=True)
    d.text(2.742,.635,"Independent weights",8.5,color=MUTED)
    d.text(3.940,.478,"Time pooling",9.2,bold=True)
    for y,k in ((.816,7),(1.190,15),(1.564,31)):
        d.line((2.170,y),(2.309,y),arrow=True)
        d.node(2.326,y,.596,.222,f"k = {k}",TEMPORAL,size=9.2,bold=True,edge=BLUE)
        d.line((2.938,y),(3.082,y),arrow=True)
        d.grid(3.105,y)
        d.line((3.452,y),(3.588,y),arrow=True)
        d.node(3.605,y,.670,.222,"Mean+SD",FEATURES,size=8.9,edge=AMBER)
        d.line((4.288,y),(4.402,y),arrow=True)
        d.grid(4.422,y,.292,.088,5,1,False,FEATURES,AMBER)
        d.line((4.726,y),(4.843,y),arrow=True)
    d.node(4.858,1.190,.318,1.004,"Concatenate",FEATURES,size=8.9,rotation=90,bold=True,edge=AMBER)
    d.line((5.192,1.190),(5.325,1.190),(5.325,.629),(5.466,.629),arrow=True)
    for y,label in ((.629,"192 → 96\nGN8 + GELU"),(.995,"96 → 96\nGN8 + GELU"),
                    (1.361,"96 → 2\nSoftmax"),(1.735,"P(PD)")):
        d.node(5.482,y,1.395,.295 if y < 1.7 else .245,label,CLASSIFIER,
               size=8.65,edge=CORAL,bold=y>=1.361)
    for y in (.629,.995,1.361):
        d.line((6.180,y+.154),(6.180,y+.209),arrow=True)
    d.text(3.272,1.846,"32 × 64",8.55,color=MUTED)
    d.text(4.280,1.846,"64 features",8.55,color=MUTED)
    d.text(5.100,1.846,"192 features",8.6,bold=True)

    d.panel(.055, 2.115, 3.929, 2.554, "(b)  Frequency-band construction")
    d.rect(.165, 2.642, .796, .947, INPUT, GREEN)
    d.text(.563,2.780,"Z-scored",9.3,bold=True)
    for i in range(3):
        d.wave(.256,2.968+.176*i,.614,3.3+i*.35,1.2*i)
    d.text(.562,3.460,"3 × 512",9.0,color=MUTED)
    d.node(1.140,3.110,.427,.415,"rFFT",SPECTRAL,size=9.05,bold=True,edge=PURPLE)
    d.line((.973,3.110),(1.124,3.110),arrow=True)
    d.text(2.084,2.523,"Masks (Hz)",9.2,bold=True)
    d.line((1.581,3.110),(1.646,3.110))
    d.line((1.646,2.731),(1.646,3.731))
    d.line((2.535,2.731),(2.535,3.731))
    for y,band in zip((2.731,2.981,3.231,3.481,3.731),("1-4","4-8","8-13","13-30","30-40")):
        d.line((1.646,y),(1.742,y),arrow=True)
        d.node(1.757,y,.673,.187,band,SPECTRAL,size=8.8,edge=PURPLE)
        d.line((2.441,y),(2.535,y))
    d.node(2.671,3.231,.626,.445,"irFFT\nper band",SPECTRAL,size=8.3,bold=True,edge=PURPLE)
    d.line((2.535,3.231),(2.654,3.231),arrow=True)
    d.rect(3.412,2.642,.501,1.213,SPECTRAL,PURPLE)
    for i in range(5):
        d.wave(3.495,2.790+.155*i,.324,1.4+i*.65,.3*i,amp=.018,color="#665584")
    d.line((3.310,3.231),(3.396,3.231),arrow=True)
    d.text(3.654,3.738,"15 × 512",8.5,color=MUTED)
    d.line((3.656,3.868),(3.656,3.956),(2.911,3.956),(2.911,4.064),arrow=True)
    d.line((.563,3.601),(.563,4.256),(1.067,4.256),arrow=True)
    d.text(.717,3.906,"raw",8.5,color=MUTED)
    d.node(1.083,4.256,2.069,.347,"Concatenate + scalar z-score",FEATURES,size=8.9,bold=True,edge=AMBER)
    d.line((3.162,4.256),(3.279,4.256),arrow=True)
    d.node(3.294,4.256,.72,.31,"18 × 512",FEATURES,size=8.3,bold=True,edge=AMBER)
    d.text(2.100,4.519,"Scalar mean and SD fitted on training data",8.55,color=MUTED)

    d.panel(4.067, 2.115, 2.978, 2.554, "(c)  One temporal branch")
    for cy,label,detail in (
        (2.714,"Conv1D  18 → 36\nGN6 + GELU","k = 7 / 15 / 31\ns = 4"),
        (3.195,"Depthwise Conv1D\nGN6 + GELU","36 groups\nk = 5, s = 2"),
        (3.676,"Pointwise Conv1D\nGN8 + GELU","36 → 32\nk = 1"),
    ):
        d.node(4.208,cy,1.480,.345,label,TEMPORAL,size=8.85,bold=True,edge=BLUE)
        d.line((5.704,cy),(5.808,cy),color=EDGE,lw=.66)
        d.text(5.912,cy,detail,8.8,align="left",color=MUTED)
    for cy in (2.714,3.195):
        d.line((4.948,cy+.185),(4.948,cy+.295),arrow=True)
    d.line((4.948,3.861),(4.948,3.958),(4.555,3.958),(4.555,4.077),arrow=True)
    d.grid(4.413,4.252,.307,.212)
    d.text(4.566,4.492,"32 × 64",8.75,color=MUTED)
    d.line((4.773,4.252),(4.927,4.252),arrow=True)
    d.node(4.944,4.252,.865,.331,"Mean+SD",FEATURES,size=8.9,bold=True,edge=AMBER)
    d.line((5.825,4.252),(5.966,4.252),arrow=True)
    d.node(5.983,4.252,.894,.253,"64 features",FEATURES,size=8.85,bold=True,edge=AMBER)

    d.rect(.055,4.775,6.99,.312,CHROME)
    d.text(1.260,4.931,"Window probability",9.2,bold=True)
    d.line((2.191,4.931),(2.527,4.931),arrow=True)
    d.text(3.552,4.931,"Subject mean",9.2,bold=True)
    d.line((4.472,4.931),(4.809,4.931),arrow=True)
    d.text(5.839,4.931,"Equal-weight seed mean",9.2,bold=True)
    return d



def draw(output, formats=("pdf", "svg", "png"), dpi=600):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    diagram = build()
    diagram.verify()
    for ext in formats:
        meta = {"CreationDate": None, "ModDate": None} if ext == "pdf" else {"Date": None} if ext == "svg" else None
        diagram.fig.savefig(output / f"model_architecture.{ext}", dpi=dpi, metadata=meta)
    plt.close(diagram.fig)
    return [f"model_architecture.{ext}" for ext in formats]
