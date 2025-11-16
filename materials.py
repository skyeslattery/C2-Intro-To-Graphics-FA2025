import numpy as np  
from utils import load_image

class Material:

    def __init__(self, k_d, k_s=0., p=20., k_m=0., k_a=None, texture_filename=None, k_t=0., ior=1.0, k_e=0., k_e_texture_filename=None):
        """
        Create a new material with the given parameters.
        
        Parameters:
          k_d : (3,) -- Diffuse coefficient (color)
          k_s : (3,) or float -- Specular coefficient
          p : float -- Specular exponent (shininess)
          k_m : (3,) or float -- Mirror reflection coefficient
          k_a : (3,) -- Ambient coefficient (defaults to k_d)
          texture_filename : str -- Path to the texture file
          k_t : (3,) or float -- Transmission (refraction) coefficient
          ior : float -- Index of Refraction (1.0 for air, 1.5 for glass)
          k_e : (3,) or float -- Emissive coefficient (base glow)
          k_e_texture_filename : str -- Path to the emission texture (glow map)
        """
        self.k_d = k_d
        self.k_s = k_s
        self.p = p
        self.k_m = k_m
        self.k_a = k_a if k_a is not None else k_d
        self.k_t = k_t
        self.ior = ior
        self.k_e = k_e
        
        self.texture_map = None
        if texture_filename:
            try:
                self.texture_map = load_image(texture_filename)
            except Exception as e:
                print(f"Warning: Could not load diffuse texture: {texture_filename}")
                self.texture_map = None
        
        self.emission_map = None
        if k_e_texture_filename:
            try:
                self.emission_map = load_image(k_e_texture_filename)
            except Exception as e:
                print(f"Warning: Could not load emission texture: {k_e_texture_filename}")
                self.emission_map = None