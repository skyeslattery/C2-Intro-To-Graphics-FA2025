from utils import *
from ray import *
from cli import render

tan  = Material(k_d=vec([0.7, 0.6, 0.3]),  k_m=0.0)
blue = Material(k_d=vec([0.3, 0.3, 0.8]),  k_m=0.0)
gray = Material(k_d=vec([0.35, 0.35, 0.35]), k_m=0.0)

glass = Material(
    # tiny diffuse
    k_d=vec([0.03, 0.03, 0.03]),
    k_s=vec([0.9, 0.9, 0.9]),
    # small mirror component
    k_m=vec([0.35, 0.35, 0.35]),
    # subtle blue tint
    k_t=vec([0.92, 0.95, 1.0]),
    ior=1.33
)

light_mat = Material(
    k_e=vec([10.5, 10.5, 10.5]),
    k_d=vec([0,0,0]),
)

scene = Scene([
    Sphere(vec([-0.7,0,0]), 0.5, tan),
    Sphere(vec([0.7,0,0]), 0.5, blue),
    Sphere(vec([0,0,0]),   0.5, glass),
    Sphere(vec([0,-40,0]), 39.5, gray),
    Sphere(vec([0, 4, 2]), 1.2, light_mat)
])

camera = Camera(vec([3,1.2,5]), target=vec([0,-0.4,0]), vfov=24, aspect=16/9)
lights = []

render(camera, scene, lights)