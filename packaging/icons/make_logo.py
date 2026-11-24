"""Write the logo's SVG masters (``logo.svg`` for 32 px and up, ``logo-small.svg`` for 16 and 24 px) from the geometry
below, so its shapes and colors are edited in one place:

    .venv/Scripts/python packaging/icons/make_logo.py

The mark is "Tesserae": a rounded tile cut into regions, as a page is, outlined in ink. Its colors come from the app's
accent, Viridian, with a warm ochre against it. The small version keeps the same idea in four larger regions with
proportionally thicker lines, so it still reads as a cut tile at 16 px.
"""

from pathlib import Path

from PySide6.QtCore import QPointF, QRectF
from PySide6.QtGui import QPainterPath, QPolygonF

HERE = Path(__file__).resolve().parent
RESOURCES = HERE.parents[1] / "src" / "tessellatum" / "resources"

ACCENT = "#0D7468"  # Viridian, the light theme's accent
INK = "#1C1D20"
OCHRE = "#E9B44C"
PAPER = "#F7F5F0"


def blend(a: str, b: str, t: float) -> str:
    ca = [int(a[i:i + 2], 16) for i in (1, 3, 5)]
    cb = [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02X}" for x, y in zip(ca, cb))


TINT = blend(ACCENT, "#FFFFFF", 0.55)
SHADE = blend(ACCENT, "#000000", 0.35)
DEEP = blend(SHADE, "#000000", 0.35)

# On a 256 grid: the tile, then the regions (points, fill), then the inner boundaries drawn once each.
TILE = (8, 8, 240, 240, 52)
FULL = {
    "regions": [
        ([(8, 8), (150, 8), (118, 96), (8, 122)], TINT),
        ([(150, 8), (248, 8), (248, 110), (118, 96)], ACCENT),
        ([(8, 122), (118, 96), (140, 170), (70, 248), (8, 248)], SHADE),
        ([(118, 96), (248, 110), (248, 248), (140, 170)], OCHRE),
        ([(140, 170), (248, 248), (70, 248)], DEEP),
    ],
    "lines": [[(150, 8), (118, 96), (8, 122)], [(118, 96), (248, 110)], [(118, 96), (140, 170), (70, 248)],
              [(140, 170), (248, 248)]],
    "line_width": 7,
    "outline_width": 8,
}
SMALL = {
    "regions": [
        ([(8, 8), (128, 8), (104, 128), (8, 140)], TINT),
        ([(128, 8), (248, 8), (248, 120), (104, 128)], ACCENT),
        ([(8, 140), (104, 128), (136, 248), (8, 248)], SHADE),
        ([(104, 128), (248, 120), (248, 248), (136, 248)], OCHRE),
    ],
    "lines": [[(128, 8), (104, 128), (8, 140)], [(104, 128), (248, 120)], [(104, 128), (136, 248)]],
    "line_width": 14,
    "outline_width": 16,
}


def _tile_path(inset: float = 0.0) -> QPainterPath:
    x, y, w, h, r = TILE
    path = QPainterPath()
    path.addRoundedRect(QRectF(x + inset, y + inset, w - 2 * inset, h - 2 * inset), r - inset, r - inset)
    return path


def _d(path: QPainterPath) -> str:
    """``path`` as SVG path data, its curves kept as curves."""
    parts, i = [], 0
    while i < path.elementCount():
        e = path.elementAt(i)
        if e.type == QPainterPath.MoveToElement:
            parts.append(f"M{e.x:.2f} {e.y:.2f}")
        elif e.type == QPainterPath.LineToElement:
            parts.append(f"L{e.x:.2f} {e.y:.2f}")
        elif e.type == QPainterPath.CurveToElement:
            c2, end = path.elementAt(i + 1), path.elementAt(i + 2)
            parts.append(f"C{e.x:.2f} {e.y:.2f} {c2.x:.2f} {c2.y:.2f} {end.x:.2f} {end.y:.2f}")
            i += 2
        i += 1
    return " ".join(parts) + " Z"


def _inside(points: list[tuple[float, float]], inset: float) -> list[tuple[float, float]]:
    """A line's points, its ends moved in along it to where it crosses the tile ``inset`` in, so that its round caps
    stay under the outline instead of poking out of the tile."""
    area = _tile_path(inset)

    def pull(outer, inner):
        a, b = QPointF(*inner), QPointF(*outer)  # a inside, b maybe outside
        if area.contains(b):
            return outer
        for _ in range(40):
            mid = (a + b) / 2
            a, b = (mid, b) if area.contains(mid) else (a, mid)
        return (a.x(), a.y())

    pts = list(points)
    pts[0] = pull(pts[0], pts[1])
    pts[-1] = pull(pts[-1], pts[-2])
    return pts


def svg(design: dict, title: str) -> str:
    """The mark as plain SVG (no clipping or masks, which some renderers ignore): each region already cut to the
    rounded tile, and each boundary line ending under the outline."""
    tile = _tile_path()
    regions = []
    for points, fill in design["regions"]:
        poly = QPainterPath()
        poly.addPolygon(QPolygonF([QPointF(px, py) for px, py in points]))
        poly.closeSubpath()
        regions.append(f'  <path d="{_d(poly.intersected(tile).simplified())}" fill="{fill}"/>')
    ow, lw = design["outline_width"], design["line_width"]
    lines = []
    for points in design["lines"]:
        pts = _inside(points, ow / 2)
        lines.append('    <path d="' + " ".join(f"{'M' if i == 0 else 'L'}{x:.2f} {y:.2f}" for i, (x, y) in enumerate(pts)) + '"/>')
    outline = _tile_path(ow / 2)
    x, y, w, h, r = TILE
    nl = chr(10)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256" viewBox="0 0 256 256">{nl}'
        f"  <title>{title}</title>{nl}"
        f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{PAPER}"/>{nl}'
        + nl.join(regions) + nl
        + f'  <g fill="none" stroke="{INK}" stroke-width="{lw}" stroke-linecap="round" stroke-linejoin="round">{nl}'
        + nl.join(lines) + nl + "  </g>" + nl
        + f'  <path d="{_d(outline)}" fill="none" stroke="{INK}" stroke-width="{ow}"/>{nl}'
        + "</svg>" + nl
    )


if __name__ == "__main__":
    (RESOURCES / "logo.svg").write_text(svg(FULL, "Tessellatum"), encoding="utf-8", newline="\n")
    (RESOURCES / "logo-small.svg").write_text(svg(SMALL, "Tessellatum (small)"), encoding="utf-8", newline="\n")
    print("written logo.svg and logo-small.svg")
