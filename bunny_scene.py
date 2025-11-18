import time
from utils import *
from ray import *
from cli import render

tan = Material(vec([0.7, 0.7, 0.4]), 0.6)
gray = Material(vec([0.2, 0.2, 0.2]), k_m=0.4)

print("Loading bunny.obj...")
try:
    vs_list = 10.0 * read_obj_triangles(open("assets/bunny.obj")) + vec([0, 0.1, 0])
except FileNotFoundError:
    print("="*50)
    print("ERROR: bunny.obj not found!")
    print("Please download the Stanford Bunny OBJ file and save it as 'bunny.obj' in this folder.")
    print("You can find it by searching for 'Stanford Bunny obj'.")
    print("="*50)
    exit()

print(f"Loaded {len(vs_list)} triangles.")


surfs = [Triangle(vs, tan) for vs in vs_list]

surfs.append(Sphere(vec([0, -40.1, 0]), 39.5, gray))

scene = Scene(surfs)

lights = [
    PointLight(vec([12, 10, 5]), vec([300, 300, 300])),
    AmbientLight(0.1),
]

all_verts = np.vstack(vs_list)
bbox_min = all_verts.min(axis=0)
bbox_max = all_verts.max(axis=0)
center = (bbox_min + bbox_max) * 0.5

extents = bbox_max - bbox_min
radius = np.linalg.norm(extents) * 0.5
eye_offset = vec([0.0, extents[1] * 0.2, radius * 2.5])
eye = center + eye_offset
camera = Camera(eye, target=center, vfov=25, aspect=16/9)
camera.target = center

print("Bunny bbox min:", bbox_min, "max:", bbox_max)
print("Bunny center:", center)
print("Camera eye:", camera.eye)
print("Camera target:", camera.target)

print("Starting render...")
start_time = time.time()

render(camera, scene, lights)

end_time = time.time()
print(f"Time taken: {end_time - start_time:.2f} seconds.")