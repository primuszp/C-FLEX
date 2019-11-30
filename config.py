"""
Configurations.

Copyright (c) 2019 Haohang Huang
Licensed under the GPL License (see LICENSE for details)
Written by Haohang Huang, November 2019.
"""

class Config():
    # > MySQL database name
    DATABASE = 'erdc'

    # > Vehicle name or 1-based ID to analysis
    VEHICLE = 'Boeing 777-300' # or int

    # > No. of tires to be analyzed (-1 for full gear)
    # Note: someone argues that only a subset of tires will have superposition effect and we only need to analyze them (to reduce running time). So this number should be adjusted according to vehicle gear configuration, therefore violates our automated batch processing scheme.
    NUM_TIRES = 6

    # === Mesh Generator (2D) === #

    # === Superposition === #
    # > Pavement layer information
    SUPER_LAYERS = {
        '0' : {'Thickness': 4, 'Modulus': 300000, 'Poisson': 0.35},
        '1' : {'Thickness': 6, 'Modulus': 45000, 'Poisson': 0.35},
        '2' : {'Thickness': 5, 'Modulus': 7500, 'Poisson': 0.35}
    }

    # > Query space in X direction
    SUPER_QUERY_MESH_X = 1

    # > Query space in Y direction
    SUPER_QUERY_MESH_Y = 0.5

    # > Query space in Z direction
    SUPER_DEPTH = [0, -100]

    # > Query space in terms of tire radius in both X & Y when there is single tire only
    SUPER_QUERY_MESH_SINGLE = 10

    # > Densified region around tire location (1.5 means x-1.5r ~ x+1.5r)
    SUPER_DENSE_REGION = 1.5

    # > Scale for grid spacing in sparse region (x tire radius)
    SUPER_SPACING_SPARSE = 1

    # > Scale for grid spacing in dense region (x tire radius)
    SUPER_SPACING_DENSE = 1/5

    # > No. of grid points in Z directoin (logspace)
    SUPER_DEPTH_POINTS = 10

    # === Mesh Generator (3D) === #
