# 3D Reconstruction Project

## File Description
`notebooks/densePointCloud.ipynb` is a Jupyter notebook containing the code that was used to run the final version of the pipeline described in the report.
The other notebooks in that folder contain older, likely non-functional versions of the pipeline.

`src/project_3d_reconstruction/*` contains the rest of the backend code for the pipeline. 
`point_cloud_to_mesh.py` contains the mesh reconstruction code, while
` render_helper.py` contains a class for constructing and rendering test scenes.

## Setup

To install this project and its dependencies, run

```shell
$ pip install -e .
```

