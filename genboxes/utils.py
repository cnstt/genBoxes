import numpy as np

def box_surface(a,b,c):
    return 2*(a*b+a*c+b*c)

def sphere_surface(radius):
    return 4*np.pi*radius**2

def cylinder_surface(radius, height):
    return 2*np.pi*(radius*height + radius**2)

def check_collision(boxes, new_box, min_dist=0):
    for box in boxes:
        if not (box['x'] + box['width'] + min_dist < new_box['x'] or
                new_box['x'] + new_box['width'] + min_dist < box['x'] or
                box['y'] + box['height'] + min_dist < new_box['y'] or
                new_box['y'] + new_box['height'] + min_dist < box['y'] or
                box['z'] + box['length'] + min_dist < new_box['z'] or
                new_box['z'] + new_box['length'] + min_dist < box['z']):
            return True
    return False