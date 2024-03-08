import numpy as np
from mpl_toolkits.mplot3d import proj3d as mpl_proj3d

from project_3d_reconstruction import homography


def test_projective_transform_matches_mpl_proj3d():
    # Arbitrary test coordinates
    coords = np.arange(100 * 3).reshape(100, 3)

    # Arbitrary projective transformation
    transform = np.arange(4 * 4).reshape(4, 4)

    project_result = homography.projective_transform(coords, transform)
    mpl_result = np.array(mpl_proj3d.proj_transform(*coords.T, transform)).T

    np.testing.assert_allclose(project_result, mpl_result)
