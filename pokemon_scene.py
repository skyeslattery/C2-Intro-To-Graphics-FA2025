import time
from utils import *
# ray.py is now imported by the USE_BVH logic
# from ray import *
from cli import render
import numpy as np

USE_BVH = True

if USE_BVH:
    from ray import *
else:
    from ray_old import *


pokemon_mat = Material(
    k_d=vec([1.0, 1.0, 1.0]),
    texture_filename="grey-texture.png",
    k_s=vec([0.02, 0.02, 0.02]),  # very low specular highlights
    p=10.0,
    k_m=0.0, # no mirror reflections
    k_a=vec([1.0, 1.0, 1.0]) # Use texture for ambient
)

# ground material (gray)
gray_mat = Material(
    k_d=vec([0.62, 0.62, 0.62]), 
    k_m=0.0, # matte
    k_s=0.0, # matte
    p=10
)

# --- ENERGY BALL MATERIAL (UPDATED) ---
# This is now a "glowing, textured, glass" ball
ball_mat = Material(
    k_d=vec([0.0, 0.0, 0.0]),      # Glass has no diffuse color
    k_s=vec([0.3, 0.3, 0.3]),      # Bright white highlight
    p=200.0,                       # Very tight highlight
    k_m=0.0,                       # Set to 0. Fresnel + k_t will handle reflections.
    k_a=vec([0.0, 0.0, 0.0]),      # No ambient
    
    # EMISSION (GLOW):
    # k_e=vec([0.1, 0.1, 0.2]),      # A base blue glow
    # k_e_texture_filename="plasma-texture.png", # A wavy "plasma" texture
    
    # REFRACTION (GLASS):
    k_t=vec([0.8, 0.9, 1.0]),      # TRANSMISSION: Mostly transparent, slightly blue
    ior=1.5                       # INDEX OF REFRACTION: Bends light like water/ice
)

# Darken Pokemon Material
pokemon_mat.k_d = vec([0.003, 0.006, 0.012])
pokemon_mat.k_s = vec([0.003, 0.003, 0.003])
pokemon_mat.k_m = vec([0.0, 0.0, 0.0])


vs_list_raw, uvs_list = read_obj_triangles_with_uvs(open("Lucario2.obj"))

vs_list = 0.6 * vs_list_raw + vec([0, -0.05, -0.2])
all_verts = np.vstack(vs_list)
bbox_min = all_verts.min(axis=0)
bbox_max = all_verts.max(axis=0)
center = (bbox_min + bbox_max) * 0.5
extents = bbox_max - bbox_min
radius = np.linalg.norm(extents) * 0.5


surfs = [Triangle(vs, pokemon_mat, uvs) for vs, uvs in zip(vs_list, uvs_list)]

# ground plane (a big sphere)
surfs.append(Sphere(vec([0, -40.1, 0]), 40.0, gray_mat))

# energy ball position and size
ball_pos = vec([-0.31, 0.8, 0.04])
ball_rad = 0.1
surfs.append(Sphere(ball_pos, ball_rad, ball_mat))


scene_start_time = time.time()
scene = Scene(surfs, bg_color=vec([0.001, 0.002, 0.001]))
scene_end_time = time.time()


def make_area_lights(center, radius, total_intensity, n=3):
    lights = []
    total_intensity = np.array(total_intensity, dtype=np.float32)
    samples = n * n
    if samples <= 1:
        return [PointLight(center, total_intensity.tolist())]
        
    per_light = (total_intensity / float(samples)).tolist()
    offs = np.linspace(-radius, radius, n)
    for ix in offs:
        for iz in offs:
            pos = center + vec([ix, 0.0, iz])
            lights.append(PointLight(pos, per_light))
    return lights

# key light above model
key_center = center + vec([0.0, extents[1] * 0.9, radius * 1.0])
key_intensity = vec([0.8, 0.8, 0.7])  # total intensity for the area light
key_lights = make_area_lights(key_center, radius * 0.6, key_intensity, n=3)

# very faint rim/back light
rim_light = PointLight(center + vec([-radius * 1.0, extents[1] * 0.4, -radius * 1.5]), vec([0.08, 0.08, 0.09]))

# ball_light = PointLight(ball_pos, vec([0.2, 0.5, 1.0]))

ambient = AmbientLight(0.0)

lights = key_lights + [rim_light, ambient]

camera = Camera(
    eye= center + vec([0.0, extents[1] * 0.8 + radius * 0.2, radius * 5.0]),
    target= center + vec([0.0, extents[1] * 0.15, 0.0]),
    vfov=25, 
    aspect=16/9
)

print("Starting render...")
render_start_time = time.time()

render(camera, scene, lights)

render_end_time = time.time()
print(f"Render finished in {render_end_time - render_start_time:.2f} seconds.")

total_time = render_end_time - scene_start_time
print(f"Total time (scene init + render): {total_time:.2f} seconds.")