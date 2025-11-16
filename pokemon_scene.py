import time
from utils import *
from cli import render
import numpy as np
from ray import *


# lucario material
pokemon_mat = Material(
    k_d=vec([1.0, 1.0, 1.0]),
    texture_filename="./assets/grey-texture.png",
    k_s=vec([0.02, 0.02, 0.02]),
    p=10.0,
    k_m=0.0
)

gray_mat = Material(
    k_d=vec([0.62, 0.62, 0.62]), 
    k_m=0.0,
    k_s=0.0
)

# energy ball material
ball_mat = Material(
    k_d=vec([0.0, 0.0, 0.0]),
    k_s=vec([0.3, 0.3, 0.3]),
    texture_filename="./assets/plasma-texture.png",
    p=200.0,
    k_e=vec([0.2, 0.5, 1.0]),  # emission
    k_t=vec([0.8, 0.9, 1.0]), # transparency
    ior=1.5                   
)

# rim light
rim_light_mat = Material(
    k_e=vec([0.5, 0.5, 0.8]),
    k_d=vec([0,0,0])
)

vs_list_raw, uvs_list = read_obj_triangles_with_uvs(open("./assets/Lucario2.obj"))

vs_list = 0.6 * vs_list_raw + vec([0, -0.05, -0.2])
all_verts = np.vstack(vs_list)
bbox_min = all_verts.min(axis=0)
bbox_max = all_verts.max(axis=0)
center = (bbox_min + bbox_max) * 0.5
extents = bbox_max - bbox_min
radius = np.linalg.norm(extents) * 0.5

surfs = [Triangle(vs, pokemon_mat, uvs) for vs, uvs in zip(vs_list, uvs_list)]

# ground plane (a big sphere)
surfs.append(Sphere(vec([0, -40.0, 0]), 40.0, gray_mat))

# energy ball
ball_pos = vec([-0.31, 0.81, 0.04])
ball_rad = 0.07
surfs.append(Sphere(ball_pos, ball_rad, ball_mat))
rim_light_pos = center + vec([radius * 1.0, extents[1] * 1.5, -radius * 2.0])
surfs.append(Sphere(rim_light_pos, 0.5, rim_light_mat))

scene_start_time = time.time()
scene = Scene(
    surfs, 
    bg_color=vec([0.00, 0.00, 0.00])
)
scene_end_time = time.time()
print(f"Scene init (BVH build) took: {scene_end_time - scene_start_time:.2f}s")

camera = Camera(
    eye= center + vec([0.0, extents[1] * 0.8 + radius * 0.2, radius * 5.0]),
    target= center + vec([0.0, extents[1] * 0.15, 0.0]),
    vfov=25, 
    aspect=16/9
)

print("Starting render...")
render_start_time = time.time()

render(camera, scene, [])

render_end_time = time.time()
print(f"Render finished in {render_end_time - render_start_time:.2f} seconds.")

total_time = render_end_time - scene_start_time
print(f"Total time (scene init + render): {total_time:.2f} seconds.")