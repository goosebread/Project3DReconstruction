"""
Class to automate some of the pyrender-trimesh stuff
"""

import numpy as np
import pyrender
import trimesh
import cv2


def randomTransform(scale=1.0):
    T = np.eye(4)
    T[0:3, 3] = np.random.uniform(-1, 1, 3) * scale

    return T


def simpleTransform(x, y, z):
    T = np.eye(4)
    T[0:3, 3] = [x, y, z]
    return T


def initialCameraPose():
    c = 2 ** -0.5

    return [[1, 0, 0, 0],
            [0, c, -c, -2],
            [0, c, c, 2],
            [0, 0, 0, 1]]


class RenderHelper(object):
    def __init__(self):
        # Initialize scene
        self.scene = pyrender.Scene(ambient_light=[.1, .1, .3], bg_color=[0, 0, 0])

        self.cameraNode = None

        # Hard code light for now
        light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=2e3)
        self.scene.add(light, pose=np.eye(4))

        self.moveCamera(initialCameraPose())

    def moveCamera(self, pose):
        if self.cameraNode is not None:
            self.scene.remove_node(self.cameraNode)

        self.cameraNode = self.scene.add(pyrender.PerspectiveCamera(yfov=np.pi / 3.0), pose=pose)

    def addFromTrimesh(self, trimesh_object, pose):
        mesh = pyrender.Mesh.from_trimesh(trimesh_object, smooth=False)
        self.scene.add(mesh, pose=pose)

    def addSphere(self, radius, pose, noise=0.0):
        sphere = trimesh.creation.icosphere(subdivisions=5, radius=radius)

        if noise > 0:
            sphere.vertices += noise * np.random.randn(*sphere.vertices.shape)

        self.addFromTrimesh(sphere, pose)

    def addCube(self, size, pose):
        cube = trimesh.creation.box(extents=[size, size, size])
        self.addFromTrimesh(cube, pose)

    def render(self, show_image=False, image_filename=None):
        r = pyrender.OffscreenRenderer(512, 512)
        color, _ = r.render(self.scene)
        r.delete()

        if show_image:
            cv2.imshow("render output", color)
            cv2.waitKey(0)

        if image_filename is not None:
            cv2.imwrite(image_filename, color)

        return color


def makeTestScene1():
    renderer = RenderHelper()

    for i in range(3):
        renderer.addCube(np.random.uniform(0.1, 0.4), randomTransform(1))

    for i in range(5):
        renderer.addSphere(np.random.uniform(0.1, 0.4), randomTransform(1), noise=np.random.uniform(0.00001, 0.001))

    renderer.render(show_image=True)


def makeTestScene2():
    """
    Attempting to recreate the matplotlib one
    """

    renderer = RenderHelper()
    renderer.addCube(0.5, simpleTransform(-0.5, -0.5, -0.5))

    for i in [-0.2, 0, 0.2]:
        renderer.addCube(0.3, simpleTransform(i, i, i))

    renderer.addCube(0.5, simpleTransform(0.5, 0.5, 0.5))

    # Render two views
    renderer.moveCamera(simpleTransform(0.1, 0, 2))
    renderer.render(show_image=True)
    renderer.moveCamera(simpleTransform(-0.1, 0, 2))
    renderer.render(show_image=True)


if __name__ == '__main__':
    makeTestScene2()
