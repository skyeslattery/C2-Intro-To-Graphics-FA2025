import numpy as np

from utils import *

"""
Core implementation of the ray tracer.  This module contains the classes (Sphere, Mesh, etc.)
that define the contents of scenes, as well as classes (Ray, Hit) and functions (shade) used in 
the rendering algorithm, and the main entry point `render_image`.

In the documentation of these classes, we indicate the expected types of arguments with a
colon, and use the convention that just writing a tuple means that the expected type is a
NumPy array of that shape.  Implementations can assume these types are preconditions that
are met, and if they fail for other type inputs it's an error of the caller.  (This might 
not be the best way to handle such validation in industrial-strength code but we are adopting
this rule to keep things simple and efficient.)
"""


class Ray:

    def __init__(self, origin, direction, start=0., end=np.inf):
        """Create a ray with the given origin and direction.

        Parameters:
          origin : (3,) -- the start point of the ray, a 3D point
          direction : (3,) -- the direction of the ray, a 3D vector (not necessarily normalized)
          start, end : float -- the minimum and maximum t values for intersections
        """
        # Convert these vectors to double to help ensure intersection
        # computations will be done in double precision
        self.origin = np.array(origin, np.float64)
        self.direction = np.array(direction, np.float64)
        self.start = start
        self.end = end

class Material:

    def __init__(self, k_d, k_s=0., p=20., k_m=0., k_a=None, texture_filename=None, k_t=0., ior=1.0, k_e=0., k_e_texture_filename=None):
        """
        Create a new material with the given parameters.
        
        Parameters:
          k_d : (3,) -- Diffuse coefficient (color)
          k_s : (3,) or float -- Specular coefficient
          p : float -- Specular exponent (shininess)
          k_m : (3,) or float -- Mirror reflection coefficient
          k_a : (3,) -- Ambient coefficient (defaults to k_d)
          texture_filename : str -- Path to the texture file
          k_t : (3,) or float -- Transmission (refraction) coefficient
          ior : float -- Index of Refraction (1.0 for air, 1.5 for glass)
          k_e : (3,) or float -- Emissive coefficient (base glow)
          k_e_texture_filename : str -- Path to the emission texture (glow map)
        """
        self.k_d = k_d
        self.k_s = k_s
        self.p = p
        self.k_m = k_m
        self.k_a = k_a if k_a is not None else k_d
        self.k_t = k_t
        self.ior = ior
        self.k_e = k_e
        
        self.texture_map = None
        if texture_filename:
            try:
                self.texture_map = load_image(texture_filename)
            except Exception as e:
                print(f"Warning: Could not load diffuse texture: {texture_filename}")
                self.texture_map = None
        
        self.emission_map = None
        if k_e_texture_filename:
            try:
                self.emission_map = load_image(k_e_texture_filename)
            except Exception as e:
                print(f"Warning: Could not load emission texture: {k_e_texture_filename}")
                self.emission_map = None

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

class BVHNode:
    def __init__(self, bbox, left=None, right=None, objects=None):
        """Create a BVH node."""
        self.bbox = bbox
        self.left = left
        self.right = right
        self.objects = objects

    def intersect(self, ray):
        """Intersect the ray with the BVH tree."""
        if not self.bbox.intersect(ray):
            return no_hit

        if self.objects is not None:
            closest_hit = no_hit
            for obj in self.objects:
                hit = obj.intersect(ray)
                if hit.t < closest_hit.t and hit.t > ray.start:
                    closest_hit = hit
            return closest_hit

        hit_left = self.left.intersect(ray)
        hit_right = self.right.intersect(ray)

        if hit_left.t < hit_right.t:
            return hit_left
        else:
            return hit_right
        
    def any_intersect(self, ray):
        """Return True if any object intersects ray within [start,end]. Fast early exit."""
        if not self.bbox.intersect(ray):
            return False

        if self.objects is not None:
            for obj in self.objects:
                hit = obj.intersect(ray)
                if hit.t < np.inf and hit.t > ray.start and hit.t < ray.end:
                    return True
            return False

        if self.left is not None and self.left.any_intersect(ray):
            return True
        if self.right is not None and self.right.any_intersect(ray):
            return True
        return False

def get_bbox_center(obj):
    """Helper to get the center of an object's AABB."""
    bbox = obj.get_bbox()
    return (bbox.min + bbox.max) * 0.5

def build_bvh(objects):
    """Recursively build the BVH tree."""
    
    # Base Cases
    if len(objects) == 0:
        return BVHNode(AABB(vec([0,0,0]), vec([0,0,0])))
    
    if len(objects) == 1:
        obj = objects[0]
        return BVHNode(bbox=obj.get_bbox(), objects=[obj])
    
    if len(objects) == 2:
        left_child = build_bvh([objects[0]])
        right_child = build_bvh([objects[1]])
        
        total_bbox = AABB()
        total_bbox.grow_box(left_child.bbox)
        total_bbox.grow_box(right_child.bbox)
        
        return BVHNode(bbox=total_bbox, left=left_child, right=right_child)

    
    total_bbox = AABB()
    centers = []
    for obj in objects:
        bbox = obj.get_bbox()
        total_bbox.grow_box(bbox)
        centers.append((bbox.min + bbox.max) * 0.5)

    extents = total_bbox.max - total_bbox.min
    split_axis = np.argmax(extents)

    zipped = sorted(zip(objects, centers), key=lambda item: item[1][split_axis])
    objects, centers = zip(*zipped)
    
    mid = len(objects) // 2
    left_objects = objects[:mid]
    right_objects = objects[mid:]

    left_child = build_bvh(left_objects)
    right_child = build_bvh(right_objects)

    return BVHNode(bbox=total_bbox, left=left_child, right=right_child)

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

class Camera:

    def __init__(self, eye=vec([0,0,0]), target=vec([0,0,-1]), up=vec([0,1,0]), 
                 vfov=90.0, aspect=1.0):
        """Create a camera with given viewing parameters.

        Parameters:
          eye : (3,) -- the camera's location, aka viewpoint (a 3D point)
          target : (3,) -- where the camera is looking: a 3D point that appears centered in the view
          up : (3,) -- the camera's orientation: a 3D vector that appears straight up in the view
          vfov : float -- the full vertical field of view in degrees
          aspect : float -- the aspect ratio of the camera's view (ratio of width to height)
        """
        self.eye = eye
        self.aspect = aspect
        self.vfov = vfov
        
        self.w = normalize(eye - target)
        self.u = normalize(np.cross(up, self.w))
        self.v = np.cross(self.w, self.u)

        rads = np.radians(self.vfov)

        self.img_h_half = np.tan(rads / 2.0)
        self.img_w_half = self.aspect * self.img_h_half

    def generate_ray(self, img_point):
        """Compute the ray corresponding to a point in the image.

        Parameters:
          img_point : (2,) -- a 2D point in [0,1] x [0,1], where (0,0) is the upper left
                      corner of the image and (1,1) is the lower right.
        Return:
          Ray -- The ray corresponding to that image location (not necessarily normalized)
        """
        alpha = self.img_w_half * (img_point[0] * 2.0 - 1.0)
        beta = self.img_h_half * (1.0 - img_point[1] * 2.0)

        direction = (alpha * self.u) + (beta * self.v) - self.w

        return Ray(self.eye, normalize(direction))


class PointLight:
    def __init__(self, position, intensity):
        """Create a point light at given position and with given intensity"""
        self.position = position
        self.intensity = intensity

    def illuminate(self, ray, hit, scene, k_a, k_d, k_s, p):
        """Compute the shading at a surface point due to this light."""
        light_vec_full = self.position - hit.point
        dist_sq = np.dot(light_vec_full, light_vec_full)
        
        light_vec = normalize(light_vec_full)
        dist = np.sqrt(dist_sq)
        # Increase epsilon for shadow rays to avoid self-occlusion from numerical error
        eps = 1e-4
        shadow_ray = Ray(
            origin=hit.point + eps * light_vec,
            direction=light_vec,
            start=eps,
            end=dist - eps
        )
        if scene.is_occluded(shadow_ray):
            return np.zeros(3)

        # Shading
        normal_hit = hit.normal
        view_vec = normalize(ray.origin - hit.point)
        halfway_vec = normalize(light_vec + view_vec)
        
        intensity_attenuated = self.intensity / dist_sq
        
        diffuse = max(0.0, np.dot(normal_hit, light_vec))
        specular  = k_s * (max(0.0, np.dot(normal_hit, halfway_vec)) ** p)

        return (k_d * diffuse + specular) * intensity_attenuated

class AmbientLight:

    def __init__(self, intensity):
        """Create an ambient light of given intensity

        Parameters:
          intensity (3,) or float: the intensity of the ambient light
        """
        self.intensity = intensity

    def illuminate(self, ray, hit, scene, k_a, k_d, k_s, p):
        """Compute the shading at a surface point due to this light.

        Parameters:
          ray : Ray -- the ray that hit the surface
          hit : Hit -- the hit data
          scene : Scene -- the scene, for shadow rays
          k_a, k_d, k_s, p : Material properties -- effective material properties at the hit point
        Return:
          (3,) -- the light reflected from the surface
        """
        return k_a * self.intensity


class Scene:

    def __init__(self, surfs, bg_color=vec([0.2,0.3,0.5])):
        """Create a scene containing the given objects.
        ...
        """
        self.surfs = surfs
        self.bg_color = bg_color
        
        self.lights = []
        for surf in self.surfs:
            if np.any(surf.material.k_e > 0):
                self.lights.append(surf)

        if len(self.surfs) > 0:
            self.bvh_root = build_bvh(self.surfs)

    def intersect(self, ray):
        """Computes the first (smallest t) intersection between a ray and the scene.

        Parameters:
          ray : Ray -- the ray to intersect with the scene
        Return:
          Hit -- the hit data
        """
        if self.bvh_root is None:
            return no_hit
            
        return self.bvh_root.intersect(ray)
    
    def is_occluded(self, ray):
        """Return True if any surface blocks the ray (fast boolean)."""
        if self.bvh_root is None:
            return False
        return self.bvh_root.any_intersect(ray)

MAX_DEPTH = 8 # max recursion depth
EPSILON = 1e-4 # for offsetting rays

def sample_point_on_sphere(sphere):
    """
    Generates a random point on the surface of a sphere, its normal, 
    and the probability density (PDF) of that sample w.r.t. surface area.
    """
    z = 1.0 - 2.0 * np.random.rand()
    r = np.sqrt(max(0.0, 1.0 - z*z))
    phi = 2.0 * np.pi * np.random.rand()
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    
    local_normal = vec([x, y, z]) # This is our normal
    point = sphere.center + local_normal * sphere.radius
    
    pdf_area = 1.0 / (4.0 * np.pi * sphere.radius**2)
    
    return point, local_normal, pdf_area

def get_texture_color(texture_map, uv):
    """helper to get color from a texture map given uvs."""
    if texture_map is None or uv is None:
        return None
    
    uv_u = uv[0] % 1.0
    uv_v = (1.0 - uv[1]) % 1.0 
    
    tex_h, tex_w, _ = texture_map.shape
    tex_x = int(uv_u * (tex_w - 1))
    tex_y = int(uv_v * (tex_h - 1))
    
    return texture_map[tex_y, tex_x]

def create_orthonormal_basis(n):
    """create a local coordinate system around the normal."""
    if np.abs(n[1]) < 0.999:
        a = vec([0, 1, 0])
    else:
        a = vec([1, 0, 0])
    
    u = normalize(np.cross(a, n))
    v = np.cross(n, u)
    return u, v

def sample_cosine_hemisphere(n, u, v):
    """generate a random direction in a cosine-weighted hemisphere."""
    r1 = 2.0 * np.pi * np.random.rand()
    r2 = np.random.rand()
    r2_sqrt = np.sqrt(r2)
    
    w = n
    direction = (
        u * np.cos(r1) * r2_sqrt +
        v * np.sin(r1) * r2_sqrt +
        w * np.sqrt(1.0 - r2)
    )
    return normalize(direction)

def trace_path(ray, scene, depth, is_specular=False):
    if depth >= MAX_DEPTH:
        return np.zeros(3)

    hit = scene.intersect(ray)
    if hit.t == np.inf:
        return scene.bg_color

    mat = hit.material
    x = hit.point
    n = hit.normal
    wo = -ray.direction
    nl = n if np.dot(wo, n) > 0 else -n

    # emission
    emitted = mat.k_e
    if np.any(emitted > 0):
        if depth == 0 or is_specular:
            return emitted
        else:
            return np.zeros(3)

    kd = np.mean(mat.k_d)
    km = np.mean(mat.k_m)
    kt = np.mean(mat.k_t)
    s = kd + km + kt
    if s < 1e-6:
        return emitted

    pd = kd / s
    pm = km / s
    pt = kt / s

    r = np.random.rand()

    # glass (dielectric)
    if r < pt:
        wi = ray.direction
        cos_i = np.dot(-wi, nl)
        entering = cos_i > 0

        eta_i = 1.0
        eta_t = mat.ior
        if not entering:
            eta_i, eta_t = eta_t, eta_i

        eta = eta_i / eta_t
        R0 = ((eta_i - eta_t) / (eta_i + eta_t))**2
        F = R0 + (1 - R0) * (1 - cos_i)**5
        k = 1 - eta*eta*(1 - cos_i*cos_i)

        refl = normalize(wi - 2 * np.dot(wi, nl) * nl)

        if k < 0:
            new_dir = refl
        else:
            if np.random.rand() < F: # fresnel reflection
                new_dir = refl
            else:
                refr = eta * wi + (eta*cos_i - np.sqrt(k)) * nl
                new_dir = normalize(refr)
        
        return emitted + trace_path(Ray(x + EPSILON*new_dir, new_dir), scene, depth+1, is_specular=True)

    elif r < pt + pm:
        refl = normalize(wo - 2*np.dot(wo, nl)*nl)
        return emitted + trace_path(Ray(x + EPSILON*refl, refl), scene, depth+1, is_specular=True)

    else:
        # importance sample the lights (NEE)
        direct_light = np.zeros(3)
        if len(scene.lights) > 0:
            light = scene.lights[0]
            
            # get a random point on the light source
            (light_point, light_normal, light_pdf_area) = sample_point_on_sphere(light)
            
            to_light_vec = light_point - x
            dist_sq = np.dot(to_light_vec, to_light_vec)
            dist = np.sqrt(dist_sq)
            wi = to_light_vec / dist

            shadow_ray = Ray(x + EPSILON*wi, wi, end=(dist - EPSILON*2.0))
            if not scene.is_occluded(shadow_ray):
                
                cos_at_surface = max(0.0, np.dot(nl, wi))
                cos_at_light   = max(0.0, np.dot(light_normal, -wi))

                if cos_at_surface > 0 and cos_at_light > 0:
                    G = (cos_at_surface * cos_at_light) / dist_sq
                    
                    # diffuse BRDF
                    f_r = mat.k_d / np.pi
                    L_e = light.material.k_e
                    
                    # (BRDF * Emission * Geometry) / PDF
                    direct_light = (f_r * L_e * G) / light_pdf_area

        u, v = create_orthonormal_basis(nl)
        new_dir = sample_cosine_hemisphere(nl, u, v)
        new_ray = Ray(x + EPSILON * new_dir, new_dir)

        # russian roulette
        p = max(0.2, kd)
        if depth > 3:
            if np.random.rand() > p:
                return emitted
            
            indirect_light = (mat.k_d * trace_path(new_ray, scene, depth+1, is_specular=False)) / p
        else:
            indirect_light = mat.k_d * trace_path(new_ray, scene, depth+1, is_specular=False)

        return emitted + direct_light + indirect_light


def render_image(camera, scene, lights, nx, ny):
    """
    render a ray traced image.
    """
    
    # quality settings (16 is low, 64 is mediumish)
    samples_per_pixel = 16
    
    # to kill fireflies
    clamp_value = 2.0
    
    output_image = np.zeros((ny, nx, 3), np.float32)
    
    for i in range(ny):
        print(f"rendering row {i+1}/{ny}...")
        for j in range(nx):
            
            pixel_color = np.zeros(3)
            for s in range(samples_per_pixel):
                
                # jittered anti-aliasing
                u = (j + np.random.rand()) / nx
                v = (i + np.random.rand()) / ny
                
                ray = camera.generate_ray(np.array([u, v]))
                
                path_color = trace_path(ray, scene, 0)
                pixel_color += np.clip(path_color, 0, clamp_value)
            
            avg_color = pixel_color / samples_per_pixel
            
            # gamma correction
            output_image[i, j] = np.clip(avg_color, 0, 1)**(1.0/2.2)
                
    return output_image