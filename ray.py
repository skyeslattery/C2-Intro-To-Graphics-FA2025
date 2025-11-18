import numpy as np
from materials import Material
from geometry import Sphere, Triangle, no_hit, Hit
from bvh import build_bvh
from utils import *

"""
Core implementation of the ray tracer.
"""


class Ray:

    def __init__(self, origin, direction, start=0., end=np.inf):
        """Create a ray with the given origin and direction.
        """
        self.origin = np.array(origin, np.float64)
        self.direction = np.array(direction, np.float64)
        self.start = start
        self.end = end

class Camera:

    def __init__(self, eye=vec([0,0,0]), target=vec([0,0,-1]), up=vec([0,1,0]), 
                 vfov=90.0, aspect=1.0):
        """Create a camera with given viewing parameters.
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
        eps = 1e-4
        shadow_ray = Ray(
            origin=hit.point + eps * light_vec,
            direction=light_vec,
            start=eps,
            end=dist - eps
        )
        if scene.is_occluded(shadow_ray):
            return np.zeros(3)

        # shading
        normal_hit = hit.normal
        view_vec = normalize(ray.origin - hit.point)
        halfway_vec = normalize(light_vec + view_vec)
        
        intensity_attenuated = self.intensity / dist_sq
        
        diffuse = max(0.0, np.dot(normal_hit, light_vec))
        specular  = k_s * (max(0.0, np.dot(normal_hit, halfway_vec)) ** p)

        # Use the passed-in k_d, which might be from a texture
        return (k_d * diffuse + specular) * intensity_attenuated

class AmbientLight:

    def __init__(self, intensity):
        """Create an ambient light of given intensity
        """
        self.intensity = intensity

    def illuminate(self, ray, hit, scene, k_a, k_d, k_s, p):
        """Compute the shading at a surface point due to this light.
        """
        # Use the passed-in k_a
        return k_a * self.intensity


class Scene:

    def __init__(self, surfs, bg_color=vec([0.2,0.3,0.5])):
        """Create a scene containing the given objects.
        """
        self.surfs = surfs
        self.bg_color = bg_color
        
        self.lights = []
        for surf in self.surfs:
            if np.any(surf.material.k_e > 0):
                self.lights.append(surf)

        if len(self.surfs) > 0:
            self.bvh_root = build_bvh(self.surfs)
        else:
            self.bvh_root = None # Add this line for safety

    def intersect(self, ray):
        """Computes the first (smallest t) intersection between a ray and the scene.
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

def trace_path(ray, scene, lights, depth, is_specular=False):
    """
    Traces a path, now with Next-Event Estimation (NEE)
    AND texturing
    AND support for the old 'lights' list.
    """
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
    
    em_tex = get_texture_color(mat.emission_map, hit.uv)
    if em_tex is not None:
        emitted = emitted * em_tex

    if np.any(emitted > 0):
        if depth == 0 or is_specular:
            return emitted
        else:
            return np.zeros(3)

    # 2. GET MATERIAL PROPERTIES (WITH TEXTURES)
    # -----------------------------------------------------
    
    # Get diffuse color from texture if it exists
    k_d_color = mat.k_d
    tex = get_texture_color(mat.texture_map, hit.uv)
    if tex is not None:
        k_d_color = tex * mat.k_d # Modulate base color by texture

    # Get other properties
    k_a_eff = mat.k_a if hasattr(mat, 'k_a') else k_d_color
    k_s_eff = mat.k_s
    p_eff   = mat.p
    
    # 3. EXPLICIT LIGHTS (Old PointLight/AmbientLight)
    # -----------------------------------------------------
    ext_light = np.zeros(3)
    for L in lights or []:
        if hasattr(L, 'illuminate'):
            # Pass the (potentially textured) k_d_color and k_a_eff
            ext_light += L.illuminate(ray, hit, scene, k_a_eff, k_d_color, k_s_eff, p_eff)

    # path tracing (surface interaction)
    kd = np.mean(k_d_color)
    km = np.mean(mat.k_m)
    kt = np.mean(mat.k_t)
    s = kd + km + kt
    if s < 1e-6:
        return emitted + ext_light

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

        if k < 0: # TIR
            new_dir = refl
        else:
            if np.random.rand() < F: # Fresnel reflection
                new_dir = refl
            else:
                refr = eta * wi + (eta*cos_i - np.sqrt(k)) * nl
                new_dir = normalize(refr)
        
        return emitted + ext_light + mat.k_t * trace_path(Ray(x + EPSILON*new_dir, new_dir), scene, lights, depth+1, is_specular=True)

    # mirror (specular)
    elif r < pt + pm:
        refl = normalize(wo - 2*np.dot(wo, nl)*nl)
        return emitted + ext_light + mat.k_m * trace_path(Ray(x + EPSILON*refl, refl), scene, lights, depth+1, is_specular=True)

    else:
        direct_light = np.zeros(3)
        num_lights = len(scene.lights)
        if num_lights > 0:
            light = scene.lights[np.random.randint(num_lights)]
            
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
                    
                    f_r = k_d_color / np.pi
                    L_e = light.material.k_e
                    
                    # (BRDF * Emission * Geometry * N) / PDF
                    direct_light = (f_r * L_e * G * num_lights) / light_pdf_area

        # indirect light (diffuse bounce)
        u, v = create_orthonormal_basis(nl)
        new_dir = sample_cosine_hemisphere(nl, u, v)
        new_ray = Ray(x + EPSILON * new_dir, new_dir)

        # russian roulette
        p = max(0.2, kd)
        if depth > 3:
            if np.random.rand() > p:
                return emitted + ext_light
            
            indirect_light = (k_d_color * trace_path(new_ray, scene, lights, depth+1, is_specular=False)) / p
        else:
            indirect_light = k_d_color * trace_path(new_ray, scene, lights, depth+1, is_specular=False)

        return emitted + ext_light + direct_light + indirect_light


def render_image(camera, scene, lights, nx, ny):
    """
    render a ray traced image.
    """
    
    # quality settings (16 is low, 64 is mediumish)
    samples_per_pixel = 4

    
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
                
                path_color = trace_path(ray, scene, lights, 0)
                pixel_color += np.clip(path_color, 0, clamp_value)
            
            avg_color = pixel_color / samples_per_pixel
            
            # gamma correction
            output_image[i, j] = np.clip(avg_color, 0, 1)**(1.0/2.2)
                
    return output_image