import numpy as np

class Smoother:
    """Simple exponential smoother for 2D/3D points or scalars"""
    def __init__(self, alpha=0.6):
        self.alpha = alpha
        self.state = None

    def update(self, value):
        v = np.array(value, dtype=float)
        if self.state is None:
            self.state = v
        else:
            self.state = self.alpha * v + (1 - self.alpha) * self.state
        return self.state


def angle_between(a, b):
    """Return angle in degrees between vectors a and b"""
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    an = np.linalg.norm(a)
    bn = np.linalg.norm(b)
    if an == 0 or bn == 0:
        return 0.0
    cos = np.clip(np.dot(a, b) / (an * bn), -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def project_point(pt, w, h):
    """Convert normalized MediaPipe point to image pixel coordinates"""
    return int(pt[0] * w), int(pt[1] * h)
