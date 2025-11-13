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

    def __init__(self, k_d, k_s=0., p=20., k_m=0., k_a=None):
        """Create a new material with the given parameters.

        Parameters:
          k_d : (3,) -- the diffuse coefficient
          k_s : (3,) or float -- the specular coefficient
          p : float -- the specular exponent
          k_m : (3,) or float -- the mirror reflection coefficient
          k_a : (3,) -- the ambient coefficient (defaults to match diffuse color)
        """
        self.k_d = k_d
        self.k_s = k_s
        self.p = p
        self.k_m = k_m
        self.k_a = k_a if k_a is not None else k_d


class Hit:

    def __init__(self, t, point=None, normal=None, material=None):
        """Create a Hit with the given data.

        Parameters:
          t : float -- the t value of the intersection along the ray
          point : (3,) -- the 3D point where the intersection happens
          normal : (3,) -- the 3D outward-facing unit normal to the surface at the hit point
          material : (Material) -- the material of the surface
        """
        self.t = t
        self.point = point
        self.normal = normal
        self.material = material

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

    def __init__(self, vs, material):
        """Create a triangle from the given vertices.

        Parameters:
          vs (3,3) -- an arry of 3 3D points that are the vertices (CCW order)
          material : Material -- the material of the surface
        """
        self.vs = vs
        self.material = material

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
        edge_1 = v_b - v_a
        edge_2 = v_c - v_a

        temp_vec = np.cross(ray.direction, edge_2)
        det = np.dot(edge_1, temp_vec)

        if -1e-8 < det and det < 1e-8:
            return no_hit

        inverse_det = 1.0 / det
        s = ray.origin - v_a
        u = np.dot(s, temp_vec) * inverse_det

        if u < 0 or u > 1:
            return no_hit

        temp_vec2 = np.cross(s, edge_1)
        v = np.dot(ray.direction, temp_vec2) * inverse_det

        if v < 0 or u + v > 1:
            return no_hit

        t = np.dot(edge_2, temp_vec2) * inverse_det

        if t > ray.start and t < ray.end:
            point = ray.origin + t * ray.direction
            normal = normalize(np.cross(edge_1, edge_2))
            return Hit(t, point, normal, self.material)

        return no_hit


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
        """Create a point light at given position and with given intensity

        Parameters:
          position : (3,) -- 3D point giving the light source location in scene
          intensity : (3,) or float -- RGB or scalar intensity of the source
        """
        self.position = position
        self.intensity = intensity

    def illuminate(self, ray, hit, scene):
        """Compute the shading at a surface point due to this light.

        Parameters:
          ray : Ray -- the ray that hit the surface
          hit : Hit -- the hit data
          scene : Scene -- the scene, for shadow rays
        Return:
          (3,) -- the light reflected from the surface
        """
        light_vec_full = self.position - hit.point
        dist_sq = np.dot(light_vec_full, light_vec_full)
        
        light_vec = normalize(light_vec_full)

        # part 5: shadow ray, basically just check if anything is between light and hit point
        dist = np.sqrt(dist_sq)
        eps = 1e-5
        shadow_ray = Ray(
            origin=hit.point + eps * hit.normal,  #avoid self-hit
            direction=light_vec,
            start=eps,
            end=dist - eps
        )
        blocker = scene.intersect(shadow_ray) 
        if blocker.t < np.inf:
            return np.zeros(3)

        normal_hit = hit.normal
        view_vec = normalize(ray.origin - hit.point)
        halfway_vec = normalize(light_vec + view_vec)
        
        k_d = hit.material.k_d
        k_s = hit.material.k_s
        p = hit.material.p
        
        intensity_attenuated = self.intensity / dist_sq
        
        diffuse = max(0.0, np.dot(normal_hit, light_vec))
        specular  = k_s * (max(0.0, np.dot(normal_hit, halfway_vec)) ** p)

        return (k_d + specular) * intensity_attenuated * diffuse


class AmbientLight:

    def __init__(self, intensity):
        """Create an ambient light of given intensity

        Parameters:
          intensity (3,) or float: the intensity of the ambient light
        """
        self.intensity = intensity

    def illuminate(self, ray, hit, scene):
        """Compute the shading at a surface point due to this light.

        Parameters:
          ray : Ray -- the ray that hit the surface
          hit : Hit -- the hit data
          scene : Scene -- the scene, for shadow rays
        Return:
          (3,) -- the light reflected from the surface
        """
        k_a = hit.material.k_a
        return k_a * self.intensity


class Scene:

    def __init__(self, surfs, bg_color=vec([0.2,0.3,0.5])):
        """Create a scene containing the given objects.

        Parameters:
          surfs : [Sphere, Triangle] -- list of the surfaces in the scene
          bg_color : (3,) -- RGB color that is seen where no objects appear
        """
        self.surfs = surfs
        self.bg_color = bg_color

    def intersect(self, ray):
        """Computes the first (smallest t) intersection between a ray and the scene.

        Parameters:
          ray : Ray -- the ray to intersect with the scene
        Return:
          Hit -- the hit data
        """
        closest_hit = no_hit
    
        for surf in self.surfs:
            hit = surf.intersect(ray)
            
            if hit.t < closest_hit.t and hit.t > ray.start:
                closest_hit = hit
                
        return closest_hit


MAX_DEPTH = 4

def shade(ray, hit, scene, lights, depth=0):
    """Compute shading for a ray-surface intersection.

    Parameters:
      ray : Ray -- the ray that hit the surface
      hit : Hit -- the hit data
      scene : Scene -- the scene
      lights : [PointLight or AmbientLight] -- the lights
      depth : int -- the recursion depth so far
    Return:
      (3,) -- the color seen along this ray
    When mirror reflection is being computed, recursion will only proceed to a depth
    of MAX_DEPTH, with zero contribution beyond that depth.
    """
    color = np.array([0.0, 0.0, 0.0])

    for light in lights:
        color += light.illuminate(ray, hit, scene)

    # Part 7 - if material is mirror reflective, trace a reflection ray then find intersection and color the surface
    k_m = hit.material.k_m
    if np.any(k_m > 0) and depth < MAX_DEPTH:
        hit_norm = hit.normal
        direction_norm = normalize(ray.direction)
        reflection_direction = normalize(direction_norm - 2.0 * np.dot(direction_norm, hit_norm) * hit_norm)

        eps = 1e-5
        reflection_ray = Ray(origin=hit.point + eps * hit_norm, direction=reflection_direction)
        reflection_hit = scene.intersect(reflection_ray)
        if reflection_hit.t < np.inf:
            reflection_color = shade(reflection_ray, reflection_hit, scene, lights, depth + 1)
        else:
            reflection_color = scene.bg_color

        color += np.array(k_m) * reflection_color
        
    return color


def render_image(camera, scene, lights, nx, ny):
    """Render a ray traced image.

    Parameters:
      camera : Camera -- the camera defining the view
      scene : Scene -- the scene to be rendered
      lights : Lights -- the lights illuminating the scene
      nx, ny : int -- the dimensions of the rendered image
    Returns:
      (ny, nx, 3) float32 -- the RGB image
    """
    output_image = np.zeros((ny,nx,3), np.float32)
    for i in range(ny):
        for j in range(nx):
            pixel_uv = np.array([ (j + 0.5) / nx , (i + 0.5) / ny ])
            
            ray = camera.generate_ray(pixel_uv) 
            
            hit = scene.intersect(ray)

            if hit.t < np.inf:
                color = shade(ray, hit, scene, lights, 0)
                output_image[i, j] = np.clip(color, 0, 1) 
            else:
                output_image[i, j] = scene.bg_color
                
    return output_image