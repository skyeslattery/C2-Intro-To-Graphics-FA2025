import time
from utils import *
from ray import *
from cli import render


TEXTURE_FILE = "assets/WoodFloor.png"

textured_mat = Material(k_d=vec([1, 1, 1]), k_s=0.3, p=90, k_m=0.3, texture_filename=TEXTURE_FILE)
gray = Material(vec([0.2, 0.2, 0.2]), k_m=0.4)

vs_list, uvs_list = read_obj_triangles_with_uvs(open("assets/cube.obj"))
vs_list = 0.5 * vs_list

print(f"Loaded {len(vs_list)} triangles.")
surfs = [Triangle(vs, textured_mat, uvs) for vs, uvs in zip(vs_list, uvs_list)]

surfs.append(Sphere(vec([0, -40, 0]), 39.5, gray))

# Scene
scene = Scene(surfs)

# Lights
lights = [
    PointLight(vec([12,10,5]), vec([300,300,300])),
    AmbientLight(0.1),
]

# Camera
camera = Camera(vec([3,1.7,5]), target=vec([0,0,0]), vfov=25, aspect=16/9)

print("Starting render...")
start_time = time.time()

render(camera, scene, lights)

end_time = time.time()
print(f"Render complete in: {end_time - start_time:.2f} seconds")