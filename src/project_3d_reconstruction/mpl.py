"""
Tools for working 3D plots in `matplotlib`.
"""
# Useful references:
# - https://matplotlib.org/stable/users/explain/artists/transforms_tutorial.html
# - https://stackoverflow.com/q/7821518/
# - https://stackoverflow.com/q/63027743/
# - https://stackoverflow.com/q/59794014/
# - https://stackoverflow.com/q/13662525/
# - https://stackoverflow.com/q/10389089/

from __future__ import annotations

import io
import warnings
from typing import TYPE_CHECKING

import numpy as np
from mpl_toolkits.mplot3d import proj3d

from project_3d_reconstruction import homography

if TYPE_CHECKING:
    import numpy.typing as npt
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d import Axes3D


def fig_export_rgba(fig: Figure) -> npt.NDArray[np.uint8]:
    """Export matplotlib figure as RGBA array."""
    # image_data = np.array(fig.canvas.buffer_rgba()).copy()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="rgba", dpi="figure")
    buffer.seek(0)

    image_data = np.frombuffer(buffer.getbuffer(), dtype=np.uint8)
    return image_data.reshape((*fig.canvas.get_width_height()[::-1], 4))


def axes_eye_coordinate(axes: Axes3D) -> npt.NDArray[np.float64]:
    """
    Return the given axes' eye/camera position in world coordinates.

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    >>> ax.view_init(azim=0, elev=0)  # Camera pointing in -x direction
    >>> eye = axes_eye_coordinate(ax)
    >>> np.testing.assert_allclose(eye, [9.25, 0.5, 0.5])
    """
    # Based on matplotlib internals (as of 3.8.3). May break in future versions.
    # See `mpl_toolkits.mplot3d.Axes3D.get_proj` and functions called therein.
    # https://github.com/matplotlib/matplotlib/blob/9a607294/lib/mpl_toolkits/mplot3d/axes3d.py#L1197

    box_aspect = axes._roll_to_vertical(axes._box_aspect)

    # Projective transform from world coordinates to view box relative coordinates.
    world_transform = proj3d.world_transformation(
        *axes.get_xlim3d(),
        *axes.get_ylim3d(),
        *axes.get_zlim3d(),
        pb_aspect=box_aspect,
    )

    # Coordinates of the center of the view box. Called "R" in mpl code.
    box_center = 0.5 * box_aspect

    # Eye/camera position in view box coordinates.
    eye = box_center + axes._dist * axes_eye_direction(axes)

    # Convert from view box to world coordinates.
    # TODO: replace with more stable implementation that direct inversion.
    return homography.projective_transform(eye, np.linalg.inv(world_transform))


def axes_eye_direction(axes: Axes3D) -> npt.NDArray[np.float64]:
    """
    Return a unit vector pointing towards the eye position.

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    >>> ax.view_init(azim=0, elev=0)  # Camera pointing in -x direction
    >>> axes_eye_direction(ax)
    array([1., 0., 0.])
    >>> ax.view_init(azim=45, elev=45)
    >>> axes_eye_direction(ax)
    array([0.5       , 0.5       , 0.7071...])
    """
    elev_rad = np.deg2rad(axes.elev)
    azim_rad = np.deg2rad(axes.azim)
    return axes._roll_to_vertical(  # type: ignore
        [
            np.cos(elev_rad) * np.cos(azim_rad),
            np.cos(elev_rad) * np.sin(azim_rad),
            np.sin(elev_rad),
        ]
    )


def axes_coords_data2display(
    axes: Axes, coordinates: npt.ArrayLike
) -> npt.NDArray[np.float64]:
    """
    Convert the given 2D data coordinates to their corresponding display coordinates.
    """
    coordinates = np.asarray(coordinates)
    if coordinates.ndim != 2 or coordinates.shape[-1] != 2:  # noqa: PLR2004
        raise ValueError(
            f"coordinates must be an Nx3 array, got shape {coordinates.shape}"
        )
    return axes.transData.transform(coordinates)  # type: ignore


def axes_coords_3d2display(
    axes: Axes3D, coordinates: npt.ArrayLike
) -> npt.NDArray[np.float64]:
    """
    Convert the given world 3D coordinates to their corresponding 2D display
    coordinates.
    """
    coordinates = np.asarray(coordinates)
    if coordinates.ndim != 2 or coordinates.shape[-1] != 3:  # noqa: PLR2004
        raise ValueError(
            f"coordinates must be an Nx3 array, got shape {coordinates.shape}"
        )

    # TODO: determine how to correct the transformation for axes with a title.
    if axes.get_title():
        warnings.warn(
            f"{axes_coords_3d2display.__name__} is known to produce incorrect results"
            f" when an axes has a non-empty title",
            stacklevel=2,
        )

    # TODO: determine how to correct the transformation for non-square canvases.
    fig_dims = axes.figure.canvas.get_width_height()
    if fig_dims[0] != fig_dims[1]:
        warnings.warn(
            f"{axes_coords_3d2display.__name__} is known to produce incorrect results"
            f" the figure's canvas has unequal height and width (got size {fig_dims})",
            stacklevel=2,
        )

    proj = axes.get_proj()
    ax_data_coords = homography.projective_transform(coordinates, proj)
    return axes_coords_data2display(axes, ax_data_coords[..., :-1])
