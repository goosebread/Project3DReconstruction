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
    mesh = trimesh.creation.icosphere(subdivisions=5, radius=0.5)
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


def extractClusters(pc):
    pass

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

    points = pv.wrap(pc)
    surf = points.reconstruct_surface(nbr_sz=5)
    surf = surf.clean()
    surf = surf.smooth(n_iter=20)

    # points = pv.PolyData(pc)
    # surf = points.delaunay_3d()

    pl = pv.Plotter(shape=(1, 2))
    pl.add_mesh(points)
    pl.add_title("Point Cloud of 3D Surface")
    pl.subplot(0, 1)
    pl.add_mesh(surf, color=True, show_edges=True)
    pl.add_title("Reconstructed Surface")
    pl.show()

    plt.show()


if __name__ == '__main__':
    main()
