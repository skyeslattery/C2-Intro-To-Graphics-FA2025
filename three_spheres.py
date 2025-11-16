from utils import *
from ray import *
from cli import render

tan = Material(vec([0.4, 0.4, 0.2]), k_s=0.3, p=90, k_m=0.3)
blue = Material(vec([0.2, 0.2, 0.5]), k_m=0.5)
gray = Material(vec([0.2, 0.2, 0.2]), k_m=0.4)

# simple glass material: transparent with IOR ~1.5
glass = Material(
    k_d=vec([0.0, 0.0, 0.0]),
    k_s=vec([0.5, 0.5, 0.5]),
    p=100.0,
    k_m=vec([0.0, 0.0, 0.0]),
    k_t = vec([1.0, 1.0, 1.0]),
    ior=1.3
)

scene = Scene([
    Sphere(vec([-0.7,0,0]), 0.5, tan),
    Sphere(vec([0.7,0,0]), 0.5, blue),
    # center glass ball
    Sphere(vec([0.0,0,0]), 0.5, glass),
    Sphere(vec([0,-40,0]), 39.5, gray),
])

camera = Camera(vec([3,1.2,5]), target=vec([0,-0.4,0]), vfov=24, aspect=16/9)

lights = [
    PointLight(camera.eye, vec([50,50,50])), # <-- Use camera.eye
    AmbientLight(0.1),
]

render(camera, scene, lights)