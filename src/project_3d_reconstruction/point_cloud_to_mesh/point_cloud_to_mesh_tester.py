"""
Test code
"""

import os

if os.getlogin() == "nathan":
    # THIS IS A SUPER HACK BECAUSE I HAVE BLENDER INSTALLED WEIRDLY
    path = os.environ.get("PATH", "")
    os.environ["PATH"] = path + ":/home/nathan/blender-3.0.0-linux-x64"

import numpy as np
import trimesh
import matplotlib.pyplot as plt

import open3d as o3d
from point_cloud_to_mesh import doReconstruction, intersectionOverUnion, pointsToMeshOpen3d


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


def open3dTesting():
    # Test bunny
    bunny = o3d.data.BunnyMesh()
    gt_mesh = o3d.io.read_triangle_mesh(bunny.path)
    gt_mesh.compute_vertex_normals()
    pcd = gt_mesh.sample_points_poisson_disk(3000)

    # Test sphere
    test_mesh = generateTrimesh()
    pc = generatePointCloud(test_mesh, noise=0.02)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(100)

    rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, 0.5)
    o3d.visualization.draw_geometries([pcd, rec_mesh], point_show_normal=True)

    # mesh = pointsToMeshOpen3d(pc)

    # plotMesh(mesh)


if __name__ == '__main__':
    main()
