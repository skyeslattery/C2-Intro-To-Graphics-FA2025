import numpy as np
from utils import vec
from geometry import Sphere, AABB, Hit, no_hit
from materials import Material


class CSGEvent:
    """Enter/exit of a solid along a ray."""
    def __init__(self, t, entering, hit, side=None):
        self.t = float(t)
        self.entering = bool(entering)
        self.hit = hit
        self.side = side


class CSGSolid:
    """Common base for CSG nodes."""

    material = Material(k_d=vec([0.0, 0.0, 0.0]), k_e=0.0)

    def get_bbox(self):
        raise NotImplementedError

    def _events(self, ray):
        """Return unsorted CSGEvent list for this solid."""
        raise NotImplementedError

    def contains(self, point):
        raise NotImplementedError

    def intersect(self, ray):
        """Return the first visible hit along the ray, or no_hit."""
        evs = self._events(ray)
        if not evs:
            return no_hit

        # sort here so subclasses don't need to worry about ordering
        evs.sort(key=lambda e: e.t)
        for ev in evs:
            if ev.entering and ray.start < ev.t < ray.end:
                return ev.hit

        return no_hit


class CSGPrimitive(CSGSolid):
    """
    Leaf node wrapping a primitive. currently assumes Sphere and treats it as a solid for CSG.
    """

    def __init__(self, primitive):
        self.primitive = primitive
        self.material = CSGSolid.material

    def get_bbox(self):
        return self.primitive.get_bbox()

    def contains(self, point):
        if isinstance(self.primitive, Sphere):
            d = point - self.primitive.center
            return np.dot(d, d) <= self.primitive.radius ** 2 + 1e-6
        box = self.get_bbox()
        return np.all(point >= box.min) and np.all(point <= box.max)

    def _events(self, ray):
        """Enter/exit events for a sphere intersected by a ray."""
        if not isinstance(self.primitive, Sphere):
            return []

        center = self.primitive.center
        r = self.primitive.radius

        #quadratic coeffs
        oc = ray.origin - center
        a = np.dot(ray.direction, ray.direction)
        b = 2.0 * np.dot(ray.direction, oc)
        c = np.dot(oc, oc) - r * r

        #check discriminatn for intersect
        disc = b * b - 4.0 * a * c
        if disc < 0.0:
            return []

        #t vals
        root = np.sqrt(disc)
        t0 = (-b - root) / (2.0 * a)
        t1 = (-b + root) / (2.0 * a)
        if t0 > t1:
            t0, t1 = t1, t0

        # ignore intersects if outside ray segment
        if t1 < ray.start or t0 > ray.end:
            return []

        def make_hit(t):
            p = ray.origin + t * ray.direction
            n = (p - center) / r
            return Hit(t, p, n, self.primitive.material)

        out = []
        if ray.start <= t0 <= ray.end:
            out.append(CSGEvent(t0, True, make_hit(t0), side='A'))
        if ray.start <= t1 <= ray.end:
            out.append(CSGEvent(t1, False, make_hit(t1), side='A'))
        return out


class CSGOperation(CSGSolid):
    """Binary boolean op node."""

    def __init__(self, left, right, op):
        if op not in ('union', 'intersection', 'difference'):
            raise ValueError('Unknown CSG op %r' % (op,))
        self.left = left
        self.right = right
        self.op = op
        self.material = CSGSolid.material

    def _combine(self, a_inside, b_inside):
        if self.op == 'union':
            return a_inside or b_inside
        if self.op == 'intersection':
            return a_inside and b_inside
        #difference
        return a_inside and not b_inside

    def contains(self, point):
        return self._combine(self.left.contains(point), self.right.contains(point))

    def get_bbox(self):
        box = AABB()
        box.grow_box(self.left.get_bbox())
        box.grow_box(self.right.get_bbox())
        return box

    def _events(self, ray):
        # gather child events and tag which side they came from
        all_events = []
        for side, child in (('A', self.left), ('B', self.right)):
            for ev in child._events(ray):
                all_events.append(CSGEvent(ev.t, ev.entering, ev.hit, side=side))

        if not all_events:
            return []

        # sort here
        all_events.sort(key=lambda e: e.t)

        # figure out initial inside/outside by probing slightly along ray
        probe = ray.origin + 1e-6 * ray.direction
        inside_a = self.left.contains(probe)
        inside_b = self.right.contains(probe)
        prev_inside = self._combine(inside_a, inside_b)

        result = []
        for ev in all_events:
            if ev.side == 'A':
                inside_a = ev.entering
            else:
                inside_b = ev.entering

            now_inside = self._combine(inside_a, inside_b)
            if now_inside != prev_inside:
                result.append(CSGEvent(ev.t, now_inside, self._fix_hit(ev)))
            prev_inside = now_inside

        return result

    def _fix_hit(self, ev):
        """Flip normals on the subtracted side for A \\ B."""
        h = ev.hit
        n = h.normal
        if self.op == self.DIFFERENCE and ev.side == 'B':
            n = -n
        return Hit(h.t, h.point, n, h.material, uv=h.uv)


def csg_union(a, b):
    return CSGOperation(a, b, CSGOperation.UNION)


def csg_intersection(a, b):
    return CSGOperation(a, b, CSGOperation.INTERSECTION)


def csg_difference(a, b):
    return CSGOperation(a, b, CSGOperation.DIFFERENCE)


# CSG gives ops on base shapes to make more complex shapes
# We then when a ray is cast, each base shape returns the entry and exit points
# We sort these along the ray and determine if we are inside or outside the composite shape