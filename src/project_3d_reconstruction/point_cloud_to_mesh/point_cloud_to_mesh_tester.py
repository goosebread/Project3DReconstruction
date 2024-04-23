"""
Test code
"""

import os

if os.getlogin() == "nathan":
    # THIS IS A SUPER HACK BECAUSE I HAVE BLENDER INSTALLED WEIRDLY
    path = os.environ.get("PATH", "")
    os.environ["PATH"] = path + ":/home/nathan/blender-3.0.0-linux-x64"

import numpy as np
import pyvista as pv
import trimesh
import matplotlib.pyplot as plt

from typing import List


def generateTrimesh():
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
    # mesh = trimesh.creation.box(extents=[1, 1, 1] )

    return mesh


def generatePointCloud(mesh, noise=0.0):
    v = mesh.vertices
    v += np.random.uniform(-noise, noise, v.shape)
    return v


def plotPointCloud(pc):
    x = pc[:, 0]
    y = pc[:, 1]
    z = pc[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.axis('equal')
    ax.scatter(x, y, z)


def plotMesh(mesh):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.axis('equal')
    ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:, 1], triangles=mesh.faces, Z=mesh.vertices[:, 2])
    plt.show()


def plotReconstruction(clusters: List[np.ndarray]):
    """
    This is the old version
    """

    pl = pv.Plotter(shape=(1, 2))
    pl.add_title("Point Cloud of 3D Surface")
    for cluster in clusters:
        pl.add_mesh(cluster)
    pl.subplot(0, 1)
    pl.add_title("Reconstructed Surface")
    for cluster in clusters:
        surf = pointsToSurface(cluster)
        pl.add_mesh(surf, color=True, show_edges=True)
    pl.show()


def extractClusters(pc, distance_threshold=0.1):
    """
    This isn't super efficient, and computation time scales probably well over the square of the point cloud size
    """

    clusters = []

    while pc.shape[0] > 0:
        cluster_start = pc[0]
        pc = np.delete(pc, 0, 0)

        cluster = cluster_start.copy()
        cluster = np.expand_dims(cluster, 0)

        queue = [cluster_start]

        while len(queue) > 0:
            start_point = queue.pop(0)
            point = np.expand_dims(start_point, 0)
            cluster = np.append(cluster, point, axis=0)

            delta = pc - start_point
            distance = np.linalg.norm(delta, axis=1)
            close_enough = np.where(distance < distance_threshold)[0]
            for index in close_enough:
                queue.append(pc[index])
            pc = np.delete(pc, close_enough, 0)

        clusters.append(cluster)

    return clusters


def pointsToSurface(pc: np.ndarray):
    points = pv.wrap(pc)
    surf = points.reconstruct_surface(nbr_sz=5)
    surf = surf.clean()
    surf = surf.smooth(n_iter=2)

    return surf


def clustersToMesh(clusters: List[np.ndarray]):
    """
    Takes in a list of point cloud clusters and outputs a single object
    """

    mesh_components = []

    for cluster in clusters:
        surf = pointsToSurface(cluster)
        faces_as_array = surf.faces.reshape((surf.n_faces_strict, 4))[:, 1:]
        mesh = trimesh.Trimesh(surf.points, faces_as_array)
        trimesh.repair.fix_inversion(mesh)  # Somehow the normals point inwards sometimes
        mesh_components.append(mesh)

    return trimesh.util.concatenate(mesh_components)


def doReconstruction(pc: np.ndarray, distance_threshold=0.5):
    """
    The full pipeline
    """

    # Extract clusters
    clusters = extractClusters(pc, distance_threshold=distance_threshold)

    # Do the mesh reconstruction
    reconstructed_mesh = clustersToMesh(clusters)

    return reconstructed_mesh


def intersectionOverUnion(mesh1, mesh2, engine="blender"):
    union = trimesh.boolean.union([mesh1, mesh2], engine=engine)
    intersection = trimesh.boolean.intersection([mesh1, mesh2], engine=engine)

    return intersection.volume / union.volume


def main():
    # Generate test mesh
    mesh = generateTrimesh()
    mesh.apply_translation([0, 1.5, 0])
    mesh2 = generateTrimesh()
    mesh2.apply_translation([0, 0, 1.5])
    mesh3 = generateTrimesh()
    mesh3.apply_translation([1.5, 0, 0])
    combined = trimesh.util.concatenate([mesh, mesh2, mesh3])

    # plotMesh(combined)

    # Make point cloud from mesh
    pc = generatePointCloud(combined, noise=0.02)

    # Run reconstruction
    reconstructed_mesh = doReconstruction(pc)

    # Calculate IoU
    iou = intersectionOverUnion(combined, reconstructed_mesh)
    print(iou)

    # Plot it
    plotMesh(reconstructed_mesh)


if __name__ == '__main__':
    main()
