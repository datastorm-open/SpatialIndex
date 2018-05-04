''' Smallest enclosing circle - Library (Python)

Modified from:
Copyright (c) 2017 Project Nayuki
    https://www.nayuki.io/page/smallest-enclosing-circle
GNU Lesser General Public License
    see <http://www.gnu.org/licenses/>.

Data conventions:
    A point is a pair of floats (x, y).
    A circle is a triple of floats (center x, center y, radius).

Returns the smallest circle that encloses all the given points.
Runs in expected O(n) time, randomized.
Input: A sequence of pairs of floats or ints, e.g. [(0,5), (3.1,-2.7)].
Output: A triple of floats representing a circle.
Note: If 0 points are given, None is returned.
      If 1 point is given, a circle of radius 0 is returned.
'''

import math
import random


# Initially: No boundary points known
def make_circle(disks):
    # Convert to float and randomize order
    shuffled = [(float(x), float(y), float(r)) for (x, y, r) in disks]
    if any(r < 0. for (x, y, r) in shuffled):
        raise ValueError("Arguments contain negative radii.")
    random.shuffle(shuffled)

    # Progressively add points to circle or recompute circle
    circle = None
    for (i, disk) in enumerate(shuffled):
        if circle is None or not is_in_circle(circle, disk):
            circle = _make_circle_one(shuffled[:(i+1)], disk)
    return circle


# One boundary point known
def _make_circle_one(disks, odisk):
    circle = (odisk[0], odisk[1], odisk[2])
    for (i, disk) in enumerate(disks):
        if not is_in_circle(circle, disk):
            if circle[2] == odisk[2]:
                circle = make_diameter(disk, odisk)
            else:
                circle = _make_circle_two(disks[:(i+1)], disk, odisk)
    return circle


# Two boundary points known
def _make_circle_two_points(disks, odisk1, odisk2):
    circ = make_diameter(odisk1, odisk2)
    left = None
    right = None
    x1, y1 = odisk1
    x2, y2 = odisk2

    # For each point not in the two-point circle
    for disk in disks:
        if is_in_circle(circ, disk):
            continue

        # Form a circumcircle and classify it on left or right side
        cross = _cross_product(x1, y1, x2, y2, disk[0], disk[1])
        circ = make_circumcircle(odisk1, odisk2, disk)
        if circ is None:
            continue
        elif cross > 0.0 and (left is None or _cross_product(px, py, qx, qy, c[0], c[1]) > _cross_product(px, py, qx, qy, left[0], left[1])):
            left = c
        elif cross < 0.0 and (right is None or _cross_product(px, py, qx, qy, c[0], c[1]) < _cross_product(px, py, qx, qy, right[0], right[1])):
            right = c

    # Select which circle to return
    if left is None and right is None:
        return circ
    elif left is None:
        return right
    elif right is None:
        return left
    else:
        return left if (left[2] <= right[2]) else right


def make_circumcircle(p0, p1, p2):
    # Mathematical algorithm from Wikipedia: Circumscribed circle
    ax, ay = p0
    bx, by = p1
    cx, cy = p2
    ox = (min(ax, bx, cx) + max(ax, bx, cx)) / 2.0
    oy = (min(ay, by, cy) + max(ay, by, cy)) / 2.0
    ax -= ox; ay -= oy
    bx -= ox; by -= oy
    cx -= ox; cy -= oy
    d = (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by)) * 2.0
    if d == 0.0:
        return None
    x = ox + ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / d
    y = oy + ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / d
    ra = math.hypot(x - p0[0], y - p0[1])
    rb = math.hypot(x - p1[0], y - p1[1])
    rc = math.hypot(x - p2[0], y - p2[1])
    return (x, y, max(ra, rb, rc))


def make_diameter(p0, p1):
    cx = (p0[0] + p1[0]) / 2.0
    cy = (p0[1] + p1[1]) / 2.0
    r0 = math.hypot(cx - p0[0], cy - p0[1])
    r1 = math.hypot(cx - p1[0], cy - p1[1])
    return (cx, cy, max(r0, r1))


_EPS = 1 + 1e-14


def is_in_circle(c, p):
    if c is None:
        return False
    return math.hypot(p[0] - c[0], p[1] - c[1]) <= c[2] * _EPS


# Returns twice the signed area of the triangle defined by
# (x0, y0), (x1, y1), (x2, y2).
def _cross_product(x0, y0, x1, y1, x2, y2):
    return (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
