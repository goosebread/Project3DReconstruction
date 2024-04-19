"""
Test code
"""
import numpy as np

import pyvista as pv
import trimesh
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.axis('equal')


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

    ax.scatter(x, y, z)


def plotMesh(mesh):
    ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:, 1], triangles=mesh.faces, Z=mesh.vertices[:, 2])
    # plt.show()


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
    surf = surf.smooth(n_iter=20)

    return surf


def main():
    mesh = generateTrimesh()
    mesh.apply_translation([0, 1.5, 0])
    mesh2 = generateTrimesh()
    mesh2.apply_translation([0, 0, 1.5])
    mesh3 = generateTrimesh()
    mesh3.apply_translation([1.5, 0, 0])

    pc1 = generatePointCloud(mesh, noise=0.0)
    pc2 = generatePointCloud(mesh2, noise=0.0)
    pc3 = generatePointCloud(mesh3, noise=0.0)
    pc = np.concatenate((pc2, pc1, pc3))

    # Extract clusters
    clusters = extractClusters(pc, distance_threshold=0.5)

    # Plot it
    pl = pv.Plotter(shape=(1, 2))
    pl.add_title("Point Cloud of 3D Surface")
    pl.add_mesh(pc)
    pl.subplot(0, 1)
    pl.add_title("Reconstructed Surface")
    for cluster in clusters:
        surf = pointsToSurface(cluster)
        pl.add_mesh(surf, color=True, show_edges=True)
    pl.show()


if __name__ == '__main__':
    main()
