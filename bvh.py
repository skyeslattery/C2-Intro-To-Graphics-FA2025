from utils import vec
import numpy as np
from geometry import no_hit, AABB

class BVHNode:
    def __init__(self, bbox, left=None, right=None, objects=None):
        """Create a BVH node."""
        self.bbox = bbox
        self.left = left
        self.right = right
        self.objects = objects

    def intersect(self, ray):
        """Intersect the ray with the BVH tree."""
        if not self.bbox.intersect(ray):
            return no_hit

        if self.objects is not None:
            closest_hit = no_hit
            for obj in self.objects:
                hit = obj.intersect(ray)
                if hit.t < closest_hit.t and hit.t > ray.start:
                    closest_hit = hit
            return closest_hit

        hit_left = self.left.intersect(ray)
        hit_right = self.right.intersect(ray)

        if hit_left.t < hit_right.t:
            return hit_left
        else:
            return hit_right
        
    def any_intersect(self, ray):
        """Return True if any object intersects ray within [start,end]. Fast early exit."""
        if not self.bbox.intersect(ray):
            return False

        if self.objects is not None:
            for obj in self.objects:
                hit = obj.intersect(ray)
                if hit.t < np.inf and hit.t > ray.start and hit.t < ray.end:
                    return True
            return False

        if self.left is not None and self.left.any_intersect(ray):
            return True
        if self.right is not None and self.right.any_intersect(ray):
            return True
        return False

def get_bbox_center(obj):
    """Helper to get the center of an object's AABB."""
    bbox = obj.get_bbox()
    return (bbox.min + bbox.max) * 0.5

def build_bvh(objects):
    """Recursively build the BVH tree."""
    
    # Base Cases
    if len(objects) == 0:
        return BVHNode(AABB(vec([0,0,0]), vec([0,0,0])))
    
    if len(objects) == 1:
        obj = objects[0]
        return BVHNode(bbox=obj.get_bbox(), objects=[obj])
    
    if len(objects) == 2:
        left_child = build_bvh([objects[0]])
        right_child = build_bvh([objects[1]])
        
        total_bbox = AABB()
        total_bbox.grow_box(left_child.bbox)
        total_bbox.grow_box(right_child.bbox)
        
        return BVHNode(bbox=total_bbox, left=left_child, right=right_child)

    
    total_bbox = AABB()
    centers = []
    for obj in objects:
        bbox = obj.get_bbox()
        total_bbox.grow_box(bbox)
        centers.append((bbox.min + bbox.max) * 0.5)

    extents = total_bbox.max - total_bbox.min
    split_axis = np.argmax(extents)

    zipped = sorted(zip(objects, centers), key=lambda item: item[1][split_axis])
    objects, centers = zip(*zipped)
    
    mid = len(objects) // 2
    left_objects = objects[:mid]
    right_objects = objects[mid:]

    left_child = build_bvh(left_objects)
    right_child = build_bvh(right_objects)

    return BVHNode(bbox=total_bbox, left=left_child, right=right_child)