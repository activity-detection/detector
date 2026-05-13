"""Interactive cv2 tool for marking forbidden zones on a scene image.

Usage:
    uv run python -m src.tools.scene_annotator \\
        --source data/parking.mp4 --name parking_west

Keys:
    LMB    add polygon vertex
    U      undo last vertex
    ENTER  close current polygon (>=3 points) -> auto-named zone_1, zone_2, ...
    S      save scenes/<name>.json (+ <name>.jpg thumbnail with overlay)
    Q      quit (also: closing the window via [x] exits)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


WINDOW_TITLE = "scene_annotator"
ZONE_COLOR = (0, 180, 255)  # BGR
ZONE_FILL_ALPHA = 0.25
CURRENT_COLOR = (0, 255, 0)
TEXT_COLOR = (255, 255, 255)
TEXT_BG = (0, 0, 0)


def _grab_first_frame(path: str) -> np.ndarray | None:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def _draw_overlay(
    canvas: np.ndarray,
    zones: list[dict],
    current_points: list[tuple[int, int]],
) -> np.ndarray:
    out = canvas.copy()
    for zone in zones:
        pts = np.array(zone["points"], dtype=np.int32)
        overlay = out.copy()
        cv2.fillPoly(overlay, [pts], ZONE_COLOR)
        out = cv2.addWeighted(overlay, ZONE_FILL_ALPHA, out, 1 - ZONE_FILL_ALPHA, 0)
        cv2.polylines(out, [pts], isClosed=True, color=ZONE_COLOR, thickness=2)
        cx, cy = pts.mean(axis=0).astype(int)
        _draw_label(out, zone["name"], (int(cx), int(cy)))

    if current_points:
        pts = np.array(current_points, dtype=np.int32)
        if len(pts) >= 2:
            cv2.polylines(out, [pts], isClosed=False, color=CURRENT_COLOR, thickness=2)
        for p in current_points:
            cv2.circle(out, p, 4, CURRENT_COLOR, -1)

    _draw_label(out, f"zones: {len(zones)}  points: {len(current_points)}", (10, 25))
    _draw_label(
        out,
        "LMB add  U undo  ENTER close  S save  Q quit",
        (10, out.shape[0] - 15),
    )
    return out


def _draw_label(img: np.ndarray, text: str, pos: tuple[int, int]) -> None:
    (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    x, y = pos
    cv2.rectangle(img, (x - 2, y - th - 4), (x + tw + 2, y + bl), TEXT_BG, -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 1, cv2.LINE_AA)


def _save(
    out_dir: Path,
    name: str,
    source: str,
    frame_size: tuple[int, int],
    zones: list[dict],
    frame: np.ndarray,
) -> bool:
    if not zones:
        print("ERROR: no zones to save.")
        return False

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{name}.json"
    jpg_path = out_dir / f"{name}.jpg"

    if json_path.exists():
        ans = input(f"{json_path} exists. Overwrite? [y/N] ").strip().lower()
        if ans != "y":
            print("(canceled)")
            return False

    payload = {
        "source": source,
        "frame_size": list(frame_size),
        "zones": zones,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    cv2.imwrite(str(jpg_path), _draw_overlay(frame, zones, []))

    print(f"saved {json_path}")
    print(f"saved {jpg_path}")
    return True


def _window_closed() -> bool:
    """Detect [x] close on the window. Returns True once the window is gone.

    `cv2.waitKey` alone misses the close button on some backends (GTK on WSL),
    so the loop also polls WND_PROP_VISIBLE explicitly.
    """
    try:
        return cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Mark forbidden zones on a scene image.")
    parser.add_argument("--source", required=True, help="Video file or image path")
    parser.add_argument("--name", required=True, help="Output stem (without extension)")
    parser.add_argument("--out-dir", default="scenes", help="Output directory (default: scenes/)")
    args = parser.parse_args()

    frame = _grab_first_frame(args.source)
    if frame is None:
        print(f"ERROR: could not read first frame from {args.source}", file=sys.stderr)
        return 1
    h, w = frame.shape[:2]
    out_dir = Path(args.out_dir)

    zones: list[dict] = []
    current_points: list[tuple[int, int]] = []

    def on_mouse(event, x, y, flags, param):
        del flags, param
        if event == cv2.EVENT_LBUTTONDOWN:
            current_points.append((x, y))

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_TITLE, min(w, 1280), min(h, 720))
    cv2.setMouseCallback(WINDOW_TITLE, on_mouse)

    print(f"Loaded frame {w}x{h} from {args.source}")
    print("Click polygon vertices, ENTER closes, S saves, Q (or window [x]) quits.")

    while True:
        cv2.imshow(WINDOW_TITLE, _draw_overlay(frame, zones, current_points))
        key = cv2.waitKey(20) & 0xFF

        if _window_closed():
            break

        if key == ord("q"):
            break
        if key == ord("u"):
            if current_points:
                current_points.pop()
        elif key in (13, 10):  # Enter (CR / LF)
            if len(current_points) < 3:
                print(f"(need >=3 points; have {len(current_points)})")
                continue
            zone_name = f"zone_{len(zones) + 1}"
            zones.append({
                "name": zone_name,
                "policy": "forbidden",
                "points": [[int(x), int(y)] for x, y in current_points],
            })
            current_points.clear()
            print(f"added {zone_name}")
        elif key == ord("s"):
            _save(out_dir, args.name, args.source, (w, h), zones, frame)

    cv2.destroyAllWindows()
    cv2.waitKey(1)  # flush pending events so the window actually closes on WSL/GTK
    return 0


if __name__ == "__main__":
    sys.exit(main())
