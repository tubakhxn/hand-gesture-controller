import cv2
import numpy as np
from utils import project_point, angle_between

# visual constants
LINE_COLOR = (255, 255, 255)  # white
LINE_WIDTH = 1
DOT_RADIUS = 3
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _add_glow(img, draw_fn, alpha=0.15, layers=3):
    """Simple glow effect: draw thicker shapes on a transparent overlay and blend."""
    overlay = img.copy()
    for i in range(layers, 0, -1):
        temp = img.copy()
        draw_fn(temp, width=LINE_WIDTH + i * 2, color=LINE_COLOR)
        cv2.addWeighted(temp, alpha * (i / layers), overlay, 1 - alpha * (i / layers), 0, overlay)
    return overlay


def draw_skeleton(img, landmarks, t=0.0, width=LINE_WIDTH, color=LINE_COLOR, handedness=None):
    """Draw kinematic lines connecting finger joints and small dots for joints.
    landmarks: list of 21 (x,y) pixel tuples like MediaPipe hand landmarks projected to image coords
    t: time in seconds for animated accents
    """
    # finger indices from MediaPipe
    fingers = [
        [0, 1, 2, 3, 4],
        [0, 5, 6, 7, 8],
        [0, 9, 10, 11, 12],
        [0, 13, 14, 15, 16],
        [0, 17, 18, 19, 20],
    ]

    # draw lines
    for f in fingers:
        pts = [landmarks[i] for i in f]
        for i in range(len(pts) - 1):
            cv2.line(img, pts[i], pts[i+1], color, width, cv2.LINE_AA)

    # draw joints
    for p in landmarks:
        cv2.circle(img, p, DOT_RADIUS, color, -1, cv2.LINE_AA)
    # (NO extra decorative flow lines here) keep the skeleton clean and minimal


def draw_decorative_bones(img, landmarks, color=LINE_COLOR):
    """Draw subtle decorative accents along each finger bone to match reference style.
    This adds a thin parallel stroke and small leaf/ellipse ornaments between joints.
    landmarks are pixel (x,y) tuples.
    """
    fingers = [
        [0, 1, 2, 3, 4],
        [0, 5, 6, 7, 8],
        [0, 9, 10, 11, 12],
        [0, 13, 14, 15, 16],
        [0, 17, 18, 19, 20],
    ]
    # scale accents based on wrist->middle mcp distance
    try:
        w = landmarks[0]
        m = landmarks[9]
        scale_ref = max(1.0, int(np.hypot(m[0]-w[0], m[1]-w[1]) / 40.0))
    except Exception:
        scale_ref = 1

    for f in fingers:
        for i in range(1, len(f)):
            a = landmarks[f[i-1]]
            b = landmarks[f[i]]
            ax, ay = a
            bx, by = b
            vx = bx - ax
            vy = by - ay
            L = np.hypot(vx, vy)
            if L < 6:
                continue
            # perpendicular unit vector
            px = -vy / L
            py = vx / L
            # small offset for parallel stroke (scale with hand)
            off = int(4 * scale_ref)
            pa1 = (int(ax + px * off), int(ay + py * off))
            pb1 = (int(bx + px * off), int(by + py * off))
            pa2 = (int(ax - px * off), int(ay - py * off))
            pb2 = (int(bx - px * off), int(by - py * off))
            # draw the subtle parallel strokes
            cv2.line(img, pa1, pb1, color, 1, cv2.LINE_AA)
            cv2.line(img, pa2, pb2, color, 1, cv2.LINE_AA)

            # draw small leaf/ellipse at the segment midpoint
            mx = int((ax + bx) / 2)
            my = int((ay + by) / 2)
            leaf_w = max(6, int(L / 4))
            leaf_h = max(3, int(leaf_w / 3))
            angle = int(np.degrees(np.arctan2(vy, vx)))
            # main ellipse
            cv2.ellipse(img, (mx, my), (leaf_w, leaf_h), angle, 0, 360, color, 1, cv2.LINE_AA)
            # small inner dot
            cv2.circle(img, (mx, my), max(1, scale_ref), color, -1, cv2.LINE_AA)


def draw_palm_radial_ui(img, palm_center, rotation_deg, t=0.0, width=1, color=LINE_COLOR):
    """Draw a palm-anchored radial UI: circular rings + spoke lines + animated mechanical ticks"""
    cx, cy = palm_center
    # allow scale via width param hack: if width > 1 treat as scale
    scale = 1.0
    try:
        scale = float(width)
    except Exception:
        scale = 1.0
    # thickness must be integer for OpenCV drawing functions
    thickness = 1
    # The reference uses minimal central mechanics without full concentric rings.
    # Remove outer rings/spokes/ticks to match the requested cleaner look.
    # Draw only a small central marker and a couple of tiny segmented accents.
    cv2.circle(img, (cx, cy), int(18 * scale), color, 1, cv2.LINE_AA)
    # small segmented accents (not full rings)
    spin = (rotation_deg + t * 40) % 360
    for a in [-30, 30]:
        theta1 = np.deg2rad(a + spin)
        theta2 = np.deg2rad(a + 8 + spin)
        pt1 = (int(cx + np.cos(theta1) * 36 * scale), int(cy + np.sin(theta1) * 36 * scale))
        pt2 = (int(cx + np.cos(theta2) * 36 * scale), int(cy + np.sin(theta2) * 36 * scale))
        cv2.line(img, pt1, pt2, color, 1, cv2.LINE_AA)


def draw_rotation_text(img, palm_center, rotation_deg, color=LINE_COLOR):
    cx, cy = palm_center
    txt = f"rotation {int(rotation_deg)}"
    # larger, more visible rotation text
    red = (0, 0, 255)
    # small red triangle marker to the left of the text
    try:
        tri = np.array([[cx - 76, cy + 18], [cx - 60, cy + 10], [cx - 60, cy + 28]])
        cv2.fillPoly(img, [tri], red)
    except Exception:
        pass
    cv2.putText(img, txt, (cx - 56, cy + 26), FONT, 0.9, red, 2, cv2.LINE_AA)


def draw_cube_and_grid(img, anchor_point, t=0.0):
    """Draw a small orange cube and a blue grid beneath it, near the wrist area. Adds small pulsing animation."""
    ax, ay = anchor_point
    # cube (isometric-ish)
    orange = (0, 80, 255)
    blue = (255, 120, 0)
    size = 46
    pulse = 1 + 0.05 * np.sin(t * 6)
    size = int(size * pulse)
    # front square
    p1 = (ax, ay)
    p2 = (ax + size, ay)
    p3 = (ax + size, ay - size)
    p4 = (ax, ay - size)
    cv2.polylines(img, [np.array([p1, p2, p3, p4])], True, orange, 1, cv2.LINE_AA)
    # top square offset
    offset = (-14, -16)
    q1 = (p1[0] + offset[0], p1[1] + offset[1])
    q2 = (p2[0] + offset[0], p2[1] + offset[1])
    q3 = (p3[0] + offset[0], p3[1] + offset[1])
    q4 = (p4[0] + offset[0], p4[1] + offset[1])
    cv2.polylines(img, [np.array([q1, q2, q3, q4])], True, orange, 1, cv2.LINE_AA)
    # connect
    for pa, qb in zip([p1, p2, p3, p4], [q1, q2, q3, q4]):
        cv2.line(img, pa, qb, orange, 1, cv2.LINE_AA)

    # small grid beneath
    grid_origin = (ax + size + 8, ay + 14)
    gw = 5
    gh = 5
    cell = 12
    for r in range(gh):
        for c in range(gw):
            pt = (grid_origin[0] + c * cell, grid_origin[1] + r * cell)
            rect = np.array([pt, (pt[0]+cell, pt[1]), (pt[0]+cell, pt[1]+cell), (pt[0], pt[1]+cell)])
            cv2.polylines(img, [rect], True, blue, 1, cv2.LINE_AA)


def _draw_leaf_between(img, a, b, color=LINE_COLOR):
    """Draw a small leaf/ellipse between two points to mimic decorative links."""
    ax, ay = a
    bx, by = b
    mx = (ax + bx) // 2
    my = (ay + by) // 2
    vx = bx - ax
    vy = by - ay
    L = max(1, int(np.hypot(vx, vy)))
    # perpendicular
    px = -vy
    py = vx
    # normalize and scale
    pn = np.hypot(px, py)
    if pn == 0:
        return
    px = int(px / pn * min(12, L // 3))
    py = int(py / pn * min(12, L // 3))
    pts = np.array([[mx - px, my - py], [mx + px, my + py]])
    cv2.ellipse(img, (mx, my), (max(6, L//4), max(6, int(abs(L/6)))), int(np.degrees(np.arctan2(vy, vx))), 0, 360, color, 1, cv2.LINE_AA)


def draw_finger_leaves(img, landmarks, color=LINE_COLOR):
    # removed: decorative leaves are omitted to keep lines minimal
    return


def draw_connecting_fingertip_lines(img, landmarks, palm_center, color=LINE_COLOR):
    """Draw thin lines connecting fingertips to the palm center, as in the reference."""
    # removed: connectors to palm (kept out to match reference)
    return


def draw_palm_data_text(img, anchor_point, openness_pct, color=(0, 80, 255)):
    ax, ay = anchor_point
    txt = f"palm data\n{int(openness_pct)}%"
    # draw simple multiline by splitting
    lines = txt.split('\n')
    for i, line in enumerate(lines):
        cv2.putText(img, line, (ax - 20, ay + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        # small rectangular accent near palm anchor
        cv2.rectangle(img, (ax - 8, ay + 28), (ax + 8, ay + 36), color, 1, cv2.LINE_AA)

def draw_fingertip_gears(img, landmarks, rotation_deg, t=0.0, color=LINE_COLOR):
    """Draw small gear-like circles at fingertips that rotate with time and palm rotation."""
    tips = [4, 8, 12, 16, 20]
    for i, idx in enumerate(tips):
        cx, cy = landmarks[idx][:2]
        r = 22  # even larger fingertip gears for the reference look
        # base circle
        cv2.circle(img, (cx, cy), r, color, 1, cv2.LINE_AA)
        # animated spokes
        spokes = 6
        phase = np.deg2rad(rotation_deg + t * 180 + i * 30)
        for s in range(spokes):
            ang = phase + s * (2 * np.pi / spokes)
            x2 = int(cx + np.cos(ang) * r)
            y2 = int(cy + np.sin(ang) * r)
            cv2.line(img, (cx, cy), (x2, y2), color, 1, cv2.LINE_AA)


def landmarks_to_pixel(landmarks, w, h):
    return [(int(x * w), int(y * h)) for (x, y, z) in landmarks]
