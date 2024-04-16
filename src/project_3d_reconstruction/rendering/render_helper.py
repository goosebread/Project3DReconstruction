"""
Class to automate some of the pyrender-trimesh stuff
"""
import os.path
import random
import numpy as np
import pyrender
import trimesh
import cv2


def randomTransform(scale=1.0):
    T = np.eye(4)
    T[0:3, 3] = np.random.uniform(-1, 1, 3) * scale

    return T


def positionOnly(x, y, z):
    T = np.eye(4)
    T[0:3, 3] = [x, y, z]
    return T


def pointingAtOrigin(x, r):
    c = r ** -0.5
    #???
    #invalid rotations for c!=2
    return [[1, 0, 0, x],
            [0, c, -c, -r],
            [0, c, c, r],
            [0, 0, 0, 1]]


class RenderHelper(object):
    def __init__(self):
        # Initialize scene
        self.scene = pyrender.Scene(ambient_light=[.1, .1, .3], bg_color=[0, 0, 0])

        self.cameraNode = None

        # Hard code light for now
        light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=2e3)
        self.scene.add(light, pose=np.eye(4))

        self.moveCamera(pointingAtOrigin(0, 2))

        self.meshDict = {}

    def moveCamera(self, pose):
        if self.cameraNode is not None:
            self.scene.remove_node(self.cameraNode)

        self.cameraNode = self.scene.add(pyrender.IntrinsicsCamera(fx=2048, fy=2048, cx=1024, cy=1024), pose=pose)

    def loadFromPath(self, file_path, dict_key):
        mesh = trimesh.load(file_path, force='mesh')
        self.meshDict[dict_key] = mesh

    def addFromMeshDict(self, dict_key, pose):
        self.addFromTrimesh(self.meshDict[dict_key], pose)

    def addFromTrimesh(self, trimesh_object, pose):
        mesh = pyrender.Mesh.from_trimesh(trimesh_object, smooth=False)
        self.scene.add(mesh, pose=pose)

    def addSphere(self, radius, pose, noise=0.0):
        sphere = trimesh.creation.icosphere(subdivisions=5, radius=radius)

        if noise > 0:
            sphere.vertices += noise * np.random.randn(*sphere.vertices.shape)

        self.addFromTrimesh(sphere, pose)

    def addCube(self, size, pose, color=None):
        if color is None:
            color = [255, 255, 255]

        # color = np.random.randint(0, 255, (12, 3))

        cube = trimesh.creation.box(extents=[size, size, size], face_colors=color)
        self.addFromTrimesh(cube, pose)

    def render(self, show_image=False, image_filename=None):
        r = pyrender.OffscreenRenderer(2048, 2048)
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
    renderer.addCube(0.5, positionOnly(-0.5, -0.5, -0.5), color=[255, 0, 0])

    for i in [-0.2, 0, 0.2]:
        renderer.addCube(0.3, positionOnly(i, i, i), color=[0, 255, 0])

    renderer.addCube(0.5, positionOnly(0.5, 0.5, 0.5), color=[0, 0, 255])

    # Render two views
    renderer.moveCamera(pointingAtOrigin(-0.1, 2))
    renderer.render(show_image=True, image_filename="test1.png")
    renderer.moveCamera(pointingAtOrigin(0.1, 2))
    renderer.render(show_image=True, image_filename="test2.png")


FILE_PATH = os.path.abspath(os.path.join(os.path.abspath(__file__), "..", "..", "..", "..", "mesh_files"))
OBJECTS = [os.path.join(FILE_PATH, "coral_2", "untitled.dae"),
           os.path.join(FILE_PATH, "coral_4", "untitled.dae"),
           os.path.join(FILE_PATH, "rock_1", "untitled.dae"),
           os.path.join(FILE_PATH, "braincoral_1", "untitled.dae"),
           ]


def makeTestScene3():
    renderer = RenderHelper()

    # Load in mesh files
    names = []
    for file_path in OBJECTS:
        print(f"Loading {file_path}")
        name = file_path.split("\\")[-2]#warning slash is different for linux vs windows
        names.append(name)
        renderer.loadFromPath(file_path, name)
    print("Done loading")

    # Place them in the world
    for i in range(30):
        object_name = random.choice(names)
        renderer.addFromMeshDict(object_name, randomTransform(1))

    # Add some cubes
    for i in range(10):
        renderer.addCube(np.random.uniform(0.1, 0.4), randomTransform(1), color=np.random.randint(0, 180, 3))

    # Render two views
    renderer.moveCamera(pointingAtOrigin(-0.05, 2))
    renderer.render(show_image=True, image_filename="test1.png")
    renderer.moveCamera(pointingAtOrigin(0.05, 2))
    renderer.render(show_image=True, image_filename="test2.png")


if __name__ == '__main__':
    makeTestScene3()
