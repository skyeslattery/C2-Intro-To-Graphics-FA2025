import numpy as np
from utils import vec, normalize

class Hit:
    def __init__(self, t, point=None, normal=None, material=None, uv=None):
        """Create a Hit with the given data.

        Parameters:
          t : float -- the t value of the intersection along the ray
          point : (3,) -- the 3D point where the intersection happens
          normal : (3,) -- the 3D outward-facing unit normal to the surface at the hit point
          material : (Material) -- the material of the surface
          uv : (2,) -- the interpolated texture coordinate at the hit point
        """
        self.t = t
        self.point = point
        self.normal = normal
        self.material = material
        self.uv = uv

# Value to represent absence of an intersection
no_hit = Hit(np.inf)


class Sphere:

    def __init__(self, center, radius, material):
        """Create a sphere with the given center and radius.

        Parameters:
          center : (3,) -- a 3D point specifying the sphere's center
          radius : float -- a Python float specifying the sphere's radius
          material : Material -- the material of the surface
        """
        self.center = center
        self.radius = radius
        self.material = material

    def get_bbox(self):
        """Return the AABB for this sphere."""
        r_vec = vec([self.radius, self.radius, self.radius])
        return AABB(self.center - r_vec, self.center + r_vec)

    def intersect(self, ray):
        """Computes the first (smallest t) intersection between a ray and this sphere.

        Parameters:
          ray : Ray -- the ray to intersect with the sphere
        Return:
          Hit -- the hit data
        """
        sphere_vec = ray.origin - self.center
        a = np.dot(ray.direction, ray.direction)
        b = 2 * np.dot(ray.direction, sphere_vec)
        c = np.dot(sphere_vec, sphere_vec) - self.radius * self.radius
        discriminant = b * b - 4 * a * c
        if discriminant < 0:
            return no_hit
        else:
            disc_sqrt = np.sqrt(discriminant)
            minus = (-b - disc_sqrt) / (2 * a)
            plus = (-b + disc_sqrt) / (2 * a)
            hit = None
            if ray.start < minus and minus < ray.end:
                hit = minus
            elif ray.start < plus and plus < ray.end:
                hit = plus
            if hit is not None:
                point = ray.origin + hit * ray.direction
                normal = (point - self.center) / self.radius
                return Hit(hit, point, normal, self.material)
        return no_hit


class Triangle:

    def __init__(self, vs, material, uvs=None):
        """Create a triangle from the given vertices.

        Parameters:
          vs (3,3) -- an arry of 3 3D points that are the vertices (CCW order)
          material : Material -- the material of the surface
          uvs (3,2) -- an array of 3 2D texture coordinates for the vertices
        """
        self.vs = vs
        self.material = material
        self.uvs = uvs
        self.edge_1 = self.vs[1] - self.vs[0]
        self.edge_2 = self.vs[2] - self.vs[0]
        self.normal = normalize(np.cross(self.edge_1, self.edge_2))

    def get_bbox(self):
        """Return the AABB for this triangle."""
        bbox = AABB()
        bbox.grow(self.vs[0])
        bbox.grow(self.vs[1])
        bbox.grow(self.vs[2])
        return bbox

    def intersect(self, ray):
        """Computes the intersection between a ray and this triangle, if it exists.

        Parameters:
          ray : Ray -- the ray to intersect with the triangle
        Return:
          Hit -- the hit data
        """
        v_a = self.vs[0]
        v_b = self.vs[1]
        v_c = self.vs[2]
        temp_vec = np.cross(ray.direction, self.edge_2)
        det = np.dot(self.edge_1, temp_vec)

        if -1e-8 < det and det < 1e-8:
            return no_hit

        inverse_det = 1.0 / det
        s = ray.origin - v_a
        u = np.dot(s, temp_vec) * inverse_det

        if u < 0 or u > 1:
            return no_hit

        temp_vec2 = np.cross(s, self.edge_1)
        v = np.dot(ray.direction, temp_vec2) * inverse_det

        if v < 0 or u + v > 1:
            return no_hit

        t = np.dot(self.edge_2, temp_vec2) * inverse_det

        if t > ray.start and t < ray.end:
            point = ray.origin + t * ray.direction
            
            hit_uv = None
            if self.uvs is not None:
                w_bary = 1.0 - u - v
                u_bary = v
                v_bary = u
                hit_uv = w_bary * self.uvs[0] + u_bary * self.uvs[1] + v_bary * self.uvs[2]

            return Hit(t, point, self.normal, self.material, uv=hit_uv)

        return no_hit
    
class AABB:
    def __init__(self, min_point=vec([np.inf, np.inf, np.inf]), max_point=vec([-np.inf, -np.inf, -np.inf])):
        """Create an Axis-Aligned Bounding Box."""
        self.min = min_point
        self.max = max_point

    def grow(self, point):
        """Grow the box to include a new point."""
        self.min = np.minimum(self.min, point)
        self.max = np.maximum(self.max, point)

    def grow_box(self, other_box):
        """Grow the box to include another AABB."""
        self.min = np.minimum(self.min, other_box.min)
        self.max = np.maximum(self.max, other_box.max)
        
    def get_center(self):
        """Get the center point of the AABB."""
        return (self.min + self.max) * 0.5

    def intersect(self, ray):
        """Check if the ray intersects the AABB using the 'Slab Test'."""
        tmin = ray.start
        tmax = ray.end

        for i in range(3):
            dir_i = ray.direction[i]
            orig_i = ray.origin[i]

            # If the ray is parallel to the slab, check origin against bounds
            if np.abs(dir_i) < 1e-12:
                if orig_i < self.min[i] or orig_i > self.max[i]:
                    return False
                else:
                    continue

            inv = 1.0 / dir_i
            t0 = (self.min[i] - orig_i) * inv
            t1 = (self.max[i] - orig_i) * inv

            if t0 > t1:
                t0, t1 = t1, t0

            tmin = max(tmin, t0)
            tmax = min(tmax, t1)

            if tmin > tmax:
                return False

        return True