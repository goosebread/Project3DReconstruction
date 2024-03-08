import math

import matplotlib.pyplot as plt
import numpy as np

import project_3d_reconstruction.mpl as proj_mpl


def test_axes_coords_data2display() -> None:
    # Generate random 3D datapoints.
    rng = np.random.default_rng()
    points = rng.uniform(size=(20, 2))

    # Make sure non-square canvas works
    # Make sure multiple axes works
    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(8, 6), label="label")
    ax = axs.flat[0]

    # Plot solid blue squares
    ax.scatter(*points.T, color="b", marker="s")
    # Add additional plot elements that could potentially disrupt the transformation.
    ax.set_title("title")
    ax.set_ylabel("ylabel")
    ax.set_xlabel("xlabel")
    ax.legend()

    # Note: calling tight_layout is REQUIRED.
    # TODO: determine why; possibly make the coordinate transformation work without it
    fig.tight_layout()

    # Compute expected display coordinates
    expected_coords = proj_mpl.axes_coords_data2display(ax, points)

    # Export figure as RGBA array. Verify that it contains blue pixels
    image_data = proj_mpl.fig_export_rgba(fig)
    assert np.all(image_data == [0, 0, 255, 255], axis=-1).any()

    # Color the pixels where we expected the original points to fall in RED.
    for c_x, c_y in expected_coords:
        image_data[
            -math.ceil(c_y + 5) : -math.floor(c_y - 5),
            math.floor(c_x - 5) : math.ceil(c_x + 5),
            :,
        ] = [255, 0, 0, 255]

    # fig, ax = plt.subplots()
    # ax.imshow(image_data)
    # fig.show()
    # input()

    # Check if any of the original marker pixels are stil visible.
    blue_pixels = np.all(image_data == [0, 0, 255, 255], axis=-1)
    assert np.sum(blue_pixels) == 0


def test_axes_coords_3d2display() -> None:
    # Generate random 3D datapoints.
    rng = np.random.default_rng()
    points = rng.uniform(size=(20, 3))

    # Currently only works on SQUARE figsize.
    # Currently only works when ncols=nrows
    fig, axs = plt.subplots(
        ncols=2, nrows=2, figsize=(8, 8), label="label", subplot_kw={"projection": "3d"}
    )
    ax = axs.flat[0]

    # Plot solid blue squares
    ax.scatter(*points.T, color="b", marker="s", depthshade=False)
    # Add additional plot elements that could potentially disrupt the transformation.
    # ax.set_title("title")  # TODO: setting title currently BREAKS transformation
    ax.set_ylabel("ylabel")
    ax.set_xlabel("xlabel")
    ax.legend()

    # Note: works without tight layout?
    # TODO: determine why
    # fig.tight_layout()

    # Compute expected display coordinates
    expected_coords = proj_mpl.axes_coords_3d2display(ax, points)

    # Export figure as RGBA array. Verify that it contains blue pixels
    image_data = proj_mpl.fig_export_rgba(fig)
    assert np.all(image_data == [0, 0, 255, 255], axis=-1).any()

    # Color the pixels where we expected the original points to fall in RED.
    for c_x, c_y in expected_coords:
        image_data[
            -math.ceil(c_y + 5) : -math.floor(c_y - 5),
            math.floor(c_x - 5) : math.ceil(c_x + 5),
            :,
        ] = [255, 0, 0, 255]

    # fig, ax = plt.subplots()
    # ax.imshow(image_data)
    # fig.show()
    # input()

    # Check if any of the original marker pixels are stil visible.
    blue_pixels = np.all(image_data == [0, 0, 255, 255], axis=-1)
    assert np.sum(blue_pixels) == 0
