"""Arrange per-run PNG figures into labeled grid images.

Expects files named like:
    <function>_<param>_<x0>_<y0>.png
e.g. convex_bowl_0.001_-7.0_-4.0.png, rosenbrock_1e-05_10_10.png

Rows of the grid = sorted param values (learning/decay rate).
Cols of the grid = sorted starting points (x0, y0).

Usage:
    # One grid for a single method/function:
    python make_grid.py --dir adagrad --function convex_bowl --out grids/adagrad_convex_bowl.png

    # Every method x function grid found under figures/, written to figures/grids/:
    python make_grid.py --all --out-dir grids
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

CELL_WIDTH = 700          # px, each panel is resized to this width (aspect preserved)
LABEL_FONT_SIZE = 26
TITLE_FONT_SIZE = 34
ROW_LABEL_WIDTH = 160     # left margin for row labels
COL_LABEL_HEIGHT = 60     # top margin (above grid, below title) for column labels
TITLE_HEIGHT = 60
MARGIN = 12
BG_COLOR = (255, 255, 255)
TEXT_COLOR = (20, 20, 20)


def _font(size: int) -> ImageFont.ImageFont:
    for candidate in (r"C:\Windows\Fonts\segoeui.ttf", r"C:\Windows\Fonts\arial.ttf"):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def _try_float(s: str) -> float:
    try:
        return float(s)
    except ValueError:
        return float("nan")


def discover_runs(directory: Path) -> dict[str, dict[tuple[str, str], Path]]:
    """Group PNGs in `directory` by function name.
    This really reslies on the output runs having an ordered stsructure.
    Returns {function_name: {(param, start_key): path}}.
    """
    groups: dict[str, dict[tuple[str, str], Path]] = {}
    for path in sorted(directory.glob("*.png")):
        stem = path.stem
        tokens = stem.split("_")
        if len(tokens) < 4:
            continue  # not enough tokens for function_param_x_y
        param, x0, y0 = tokens[-3], tokens[-2], tokens[-1]
        function = "_".join(tokens[:-3])
        groups.setdefault(function, {})[(param, f"{x0}_{y0}")] = path
    return groups


def build_grid(runs: dict[tuple[str, str], Path], title: str) -> Image.Image:
    params = sorted({p for p, _ in runs}, key=_try_float)
    starts = sorted({s for _, s in runs})

    n_rows, n_cols = len(params), len(starts)

    # Determine cell height from the first available image's aspect ratio.
    sample = Image.open(next(iter(runs.values())))
    cell_height = int(CELL_WIDTH * sample.height / sample.width)

    grid_w = ROW_LABEL_WIDTH + n_cols * CELL_WIDTH + (n_cols + 1) * MARGIN
    grid_h = TITLE_HEIGHT + COL_LABEL_HEIGHT + n_rows * cell_height + (n_rows + 1) * MARGIN

    canvas = Image.new("RGB", (grid_w, grid_h), BG_COLOR)
    draw = ImageDraw.Draw(canvas)

    title_font = _font(TITLE_FONT_SIZE)
    label_font = _font(LABEL_FONT_SIZE)

    draw.text((MARGIN, MARGIN), title, fill=TEXT_COLOR, font=title_font)

    for c, start in enumerate(starts):
        x0, y0 = start.split("_", 1)
        label = f"start=({x0}, {y0})"
        cx = ROW_LABEL_WIDTH + c * CELL_WIDTH + (c + 1) * MARGIN
        draw.text(
            (cx + CELL_WIDTH // 2, TITLE_HEIGHT + COL_LABEL_HEIGHT // 2),
            label,
            fill=TEXT_COLOR,
            font=label_font,
            anchor="mm",
        )

    for r, param in enumerate(params):
        cy = TITLE_HEIGHT + COL_LABEL_HEIGHT + r * cell_height + (r + 1) * MARGIN
        draw.text(
            (ROW_LABEL_WIDTH // 2, cy + cell_height // 2),
            f"rate={param}",
            fill=TEXT_COLOR,
            font=label_font,
            anchor="mm",
        )

    for r, param in enumerate(params):
        for c, start in enumerate(starts):
            path = runs.get((param, start))
            x = ROW_LABEL_WIDTH + c * CELL_WIDTH + (c + 1) * MARGIN
            y = TITLE_HEIGHT + COL_LABEL_HEIGHT + r * cell_height + (r + 1) * MARGIN
            if path is None:
                draw.rectangle([x, y, x + CELL_WIDTH, y + cell_height], outline=(200, 0, 0))
                draw.text((x + 10, y + 10), "missing", fill=(200, 0, 0), font=label_font)
                continue
            img = Image.open(path).convert("RGB")
            img = img.resize((CELL_WIDTH, cell_height), Image.LANCZOS)
            canvas.paste(img, (x, y))

    return canvas


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, help="Method subdirectory, e.g. adagrad")
    ap.add_argument("--function", type=str, help="Function prefix, e.g. convex_bowl")
    ap.add_argument("--out", type=Path, help="Output PNG path for a single grid")
    ap.add_argument("--all", action="store_true", help="Build grids for every method/function found under --root")
    ap.add_argument("--root", type=Path, default=Path("."), help="Root containing method subdirectories (default: cwd)")
    ap.add_argument("--out-dir", type=Path, default=Path("grids"), help="Output directory when using --all")
    args = ap.parse_args()

    if args.all:
        out_dir = args.root / args.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        for method_dir in sorted(p for p in args.root.iterdir() if p.is_dir() and p != out_dir):
            groups = discover_runs(method_dir)
            if not groups:
                continue
            for function, runs in groups.items():
                title = f"{method_dir.name} — {function}"
                grid = build_grid(runs, title)
                out_path = out_dir / f"{method_dir.name}_{function}.png"
                grid.save(out_path)
                print(f"wrote {out_path} ({grid.width}x{grid.height}, {len(runs)} panels)")
        return

    if not args.dir or not args.function or not args.out:
        ap.error("--dir, --function, and --out are required unless --all is given")

    groups = discover_runs(args.dir)
    if args.function not in groups:
        available = ", ".join(sorted(groups))
        raise SystemExit(f"function '{args.function}' not found in {args.dir}. Available: {available}")

    title = f"{args.dir.name} — {args.function}"
    grid = build_grid(groups[args.function], title)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    grid.save(args.out)
    print(f"wrote {args.out} ({grid.width}x{grid.height}, {len(groups[args.function])} panels)")


if __name__ == "__main__":
    main()
