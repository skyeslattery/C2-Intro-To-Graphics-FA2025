from utils import *
from ray import *
from cli import render
from csg import CSGPrimitive, csg_intersection, csg_difference

#Materials defs
saucer_mat = Material(
    k_d=vec([0.7, 0.7, 0.8]),
    k_s=vec([0.3, 0.3, 0.3]),
    p=80.0,
    k_m=vec([0.1, 0.1, 0.1])
)

ground_mat = Material(
    k_d=vec([0.25, 0.25, 0.25]),
    k_m=0.0
)

light_mat = Material(
    k_e=vec([1.2, 1.2, 1.2]),
    k_d=vec([0.0, 0.0, 0.0])
)

dome_mat = Material(
    k_d=vec([0.2, 0.5, 0.7]),
    k_s=vec([0.1, 0.1, 0.1]),
    p=60.0
)

# Two large spheres to make UFO body
outer1 = Sphere(vec([0.0, -0.35, 0.0]), 1.0, saucer_mat)
outer2 = Sphere(vec([0.0,  0.35, 0.0]), 1.0, saucer_mat) 

body = csg_intersection(
    CSGPrimitive(outer1),
    CSGPrimitive(outer2)
)

inner_cut = Sphere(vec([0.0, 0.0, 0.0]), 0.8, saucer_mat)

#glowing cockpit dome
cockpit_sphere = Sphere(vec([0.0, 0.5, 0.0]), 0.7, dome_mat)
cockpit = CSGPrimitive(cockpit_sphere)

saucer = csg_difference(
    body,
    CSGPrimitive(inner_cut),
)

# ground+light

ground = Sphere(vec([0.0, -40.0, 0.0]), 39.5, ground_mat)
ceiling_light = Sphere(vec([0.0, 4.0, 2.0]), 1.0, light_mat)

scene = Scene([
    saucer,
    cockpit,
    ground,
    ceiling_light,
])

camera = Camera(
    eye=vec([3.0, 1.0, 6.0]),
    target=vec([0.0, -0.1, 0.0]),
    vfov=20,
    aspect=16/9
)


lights = []

render(camera, scene, lights)