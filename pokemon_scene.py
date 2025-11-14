import time
from utils import *
from ray import *
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
    k_s=vec([0.1, 0.1, 0.1]),  # specular highlights
    p=100.0
)

# ground material (gray)
gray_mat = Material(
    k_d=vec([0.12, 0.12, 0.12]), 
    k_m=vec([0.0, 0.0, 0.0]), # matte
    k_s=vec([0.01, 0.01, 0.01]),
    p=50
)

# energy ball material
ball_mat = Material(
    k_d=vec([0.0, 0.0, 0.0]),
    k_s=vec([0.8, 0.8, 0.8]),  # white highlight
    p=200.0,
    k_m=vec([0.8, 0.8, 0.8]),
    k_a=vec([20.0, 20.0, 20.0]),
    # Strong blue emissive term so the ball surface visibly glows
    # (emissive contribution is surface-only in this renderer)
    k_e=vec([0.0, 0.0, 2.5]),
    ior=1.33
)

pokemon_mat.k_d = vec([0.02, 0.04, 0.08])  # dark blue base tint
pokemon_mat.k_s = vec([0.02, 0.02, 0.02])  # very low specular highlights
pokemon_mat.k_m = vec([0.0, 0.0, 0.0])     # no mirror reflections


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

# energy ball
surfs.append(Sphere(vec([-0.31, 0.8, 0.04]), 0.05, ball_mat))


scene_start_time = time.time()
scene = Scene(surfs, bg_color=vec([0.001, 0.002, 0.001]))
scene_end_time = time.time()


def make_area_lights(center, radius, total_intensity, n=3):
    lights = []
    total_intensity = np.array(total_intensity, dtype=np.float32)
    samples = n * n
    per_light = (total_intensity / float(samples)).tolist()
    offs = np.linspace(-radius, radius, n)
    for ix in offs:
        for iz in offs:
            pos = center + vec([ix, 0.0, iz])
            lights.append(PointLight(pos, per_light))
    return lights

# key light above model
key_center = center + vec([0.0, extents[1] * 0.9, radius * 1.0])
# very dim
key_intensity = vec([0.5, 0.5, 0.4])  # total intensity for the area light
key_lights = make_area_lights(key_center, radius * 0.6, key_intensity, n=3)

# very faint rim/back light
rim_light = PointLight(center + vec([-radius * 1.0, extents[1] * 0.4, -radius * 1.5]), vec([0.08, 0.08, 0.09]))

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