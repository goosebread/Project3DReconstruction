"""
Class to automate some of the pyrender-trimesh stuff
"""

import numpy as np
import pyrender
import trimesh
import cv2


class RenderHelper(object):
    def __init__(self):
        # compose scene
        self.scene = pyrender.Scene(ambient_light=[.1, .1, .3], bg_color=[0, 0, 0])
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=2e3)

        self.scene.add(light, pose=np.eye(4))

        # Hard-code some stuff for now
        c = 2 ** -0.5
        self.scene.add(camera, pose=[[1, 0, 0, 0],
                                     [0, c, -c, -2],
                                     [0, c, c, 2],
                                     [0, 0, 0, 1]])

    def addFromTrimesh(self, trimesh_object, pose):
        mesh = pyrender.Mesh.from_trimesh(trimesh_object, smooth=False)
        self.scene.add(mesh, pose=pose)

    def addSphere(self, radius, pose):
        sphere = trimesh.creation.icosphere(subdivisions=4, radius=radius)
        self.addFromTrimesh(sphere, pose)

    def render(self, show_image=False, image_filename=None):
        r = pyrender.OffscreenRenderer(512, 512)
        color, _ = r.render(self.scene)

        if show_image:
            cv2.imshow("render output", color)
            cv2.waitKey(0)

        if image_filename is not None:
            cv2.imwrite(image_filename, color)

        return color


# This just re-does the test script
if __name__ == '__main__':
    renderer = RenderHelper()
    renderer.addSphere(0.8, np.eye(4))
    renderer.render(show_image=True)
