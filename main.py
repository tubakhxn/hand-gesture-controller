import os
import cv2
import mediapipe as mp
import numpy as np
import time
from hand_overlay import (
    draw_skeleton,
    draw_palm_radial_ui,
    draw_rotation_text,
    draw_cube_and_grid,
    draw_fingertip_gears,
    draw_palm_data_text,
    landmarks_to_pixel,
)
from utils import Smoother, angle_between

# Quiet TensorFlow / TF logger from MediaPipe
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

mp_hands = mp.solutions.hands

# smoothing objects
palm_smoother = Smoother(alpha=0.6)
rot_smoother = Smoother(alpha=0.6)
openess_smoother = Smoother(alpha=0.4)


def compute_palm_rotation(landmarks):
    """Estimate palm rotation using vector from wrist (0) to middle_finger_mcp (9)
    returns degrees of rotation around camera z (2D)
    """
    w = np.array(landmarks[0][:2])
    m = np.array(landmarks[9][:2])
    v = m - w
    angle = np.degrees(np.arctan2(v[1], v[0]))
    # normalize to 0-360
    angle = float(angle % 360)
    return angle


def run():
    cap = cv2.VideoCapture(0)
    prev = time.time()
    fps = 0.0
    with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.6) as hands:
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                now = time.time()
                dt = now - prev if now - prev > 0 else 1e-6
                prev = now
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

                h, w = frame.shape[:2]
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)

                overlay = frame.copy()
                t = now
                if results.multi_hand_landmarks:
                    hand = results.multi_hand_landmarks[0]
                    lm = [(l.x, l.y, l.z) for l in hand.landmark]
                    pix = landmarks_to_pixel(lm, w, h)

                    # compute palm center roughly as average of wrist and palm base
                    palm = ((pix[0][0] + pix[9][0]) // 2, (pix[0][1] + pix[9][1]) // 2)
                    palm = tuple(map(int, palm_smoother.update(palm)))

                    rot = compute_palm_rotation(lm)
                    rot = float(rot_smoother.update(rot))

                    # draw animated HUD
                    draw_palm_radial_ui(overlay, palm, rot, t=t)
                    draw_skeleton(overlay, pix, t=t)
                    try:
                        from hand_overlay import draw_decorative_bones
                        draw_decorative_bones(overlay, pix)
                    except Exception:
                        pass
                    draw_rotation_text(overlay, palm, rot)
                    # minimal HUD (no extra decorative leaves/connectors)

                    # cube near wrist
                    wrist_anchor = (pix[0][0] - 80, pix[0][1] + 40)
                    # estimate openness: mean distance of fingertips from palm center mapped to 0-100
                    tips = [4, 8, 12, 16, 20]
                    dists = []
                    for ti in tips:
                        tx = int(lm[ti][0] * w)
                        ty = int(lm[ti][1] * h)
                        dists.append(np.hypot(tx - palm[0], ty - palm[1]))
                    openness = 0
                    if dists:
                        mean_dist = float(np.mean(dists))
                        # dynamic calibration: closed ~ small fraction of frame, open ~ 0.55 * min(w,h)
                        closed_ref = max(12.0, min(40.0, min(w, h) * 0.04))
                        open_ref = max(60.0, min(w, h) * 0.55)
                        raw_openness = (mean_dist - closed_ref) / (open_ref - closed_ref) * 100.0
                        raw_openness = float(np.clip(raw_openness, 0.0, 100.0))
                        # smooth openness so it can still reach 0/100 but not jitter
                        openness = float(openess_smoother.update(raw_openness))
                    draw_cube_and_grid(overlay, wrist_anchor, t=t)
                    # pass scale into palm UI so fingertip gear size responds
                    draw_palm_radial_ui(overlay, palm, rot, t=t, width=1.0 + openness / 160.0)
                    draw_fingertip_gears(overlay, pix, rot, t=t)
                    draw_palm_data_text(overlay, wrist_anchor, openness)

                # HUD: FPS and instructions
                cv2.putText(overlay, f"FPS: {int(fps)}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2, cv2.LINE_AA)
                cv2.putText(overlay, "Press Esc to quit", (10, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

                cv2.imshow('Hand HUD', overlay)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break
        except KeyboardInterrupt:
            # graceful exit
            pass
        finally:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    run()
