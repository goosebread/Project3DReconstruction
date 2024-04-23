"""
Test code
"""

import numpy as np
import pyvista as pv
import trimesh
import open3d as o3d

from typing import List


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


def pointsToMeshPyVista(pc: np.ndarray):
    points = pv.wrap(pc)
    surf = points.reconstruct_surface(nbr_sz=5)
    surf = surf.clean()
    surf = surf.smooth(n_iter=2)

    faces_as_array = surf.faces.reshape((surf.n_faces_strict, 4))[:, 1:]
    mesh = trimesh.Trimesh(surf.points, faces_as_array)
    trimesh.repair.fix_inversion(mesh)  # Somehow the normals point inwards sometimes

    return mesh


def pointsToMeshOpen3d(pc: np.ndarray, method=None):
    """
    https://www.open3d.org/docs/latest/tutorial/Advanced/surface_reconstruction.html
    """

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(100)

    # Poisson seems to work the best
    if method == "ball_pivot":
        radii = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]  # Can play around with these parameters
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
    elif method == "alpha":
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, 0.5)  # Can play around with the alpha (0.5) parameter
    else:
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd=pcd, depth=5)  # Can play around with the depth parameter

    tri_mesh = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles))

    # IDK which all of these are the right one, but fix_inversions was the key before
    trimesh.repair.fix_winding(tri_mesh)
    trimesh.repair.fix_inversion(tri_mesh)
    trimesh.repair.fill_holes(tri_mesh)

    return tri_mesh


def pointsToMesh(pc: np.ndarray, method=None):
    if method is None:
        method = "open3d_poisson"

    if method == "pyvista":
        return pointsToMeshPyVista(pc)
    elif method == "open3d_poisson":
        return pointsToMeshOpen3d(pc)
    elif method == "open3d_ball_pivot":
        return pointsToMeshOpen3d(pc, "ball_pivot")
    elif method == "open3d_alpha":
        return pointsToMeshOpen3d(pc, "alpha")
    else:
        raise Exception(f"Unknown conversion type {method}")


def clustersToMesh(clusters: List[np.ndarray]):
    """
    Takes in a list of point cloud clusters and outputs a single object
    """

    mesh_components = []

    for cluster in clusters:
        mesh_components.append(pointsToMesh(cluster))

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


def intersectionOverUnion(mesh1, mesh2, engine=None):
    union = trimesh.boolean.union([mesh1, mesh2], engine=engine)
    intersection = trimesh.boolean.intersection([mesh1, mesh2], engine=engine)

    return intersection.volume / union.volume
