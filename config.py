"""
Configurations.

Copyright (c) 2019 Haohang Huang
Licensed under the GPL License (see LICENSE for details)
Written by Haohang Huang, November 2019.
"""

import numpy as np

class Config():
    # ============================== #
    # ====== Hyper Parameters ====== #
    # ============================== #
    # > MySQL database name
    DATABASE = 'erdc'

    # > Vehicle name or 1-based ID
    VEHICLE = 'Boeing 777-300' # or int

    # > No. of tires to be analyzed (-1 for full gear)
    # Note: someone argues that only a subset of tires will have superposition effect and we only need to analyze them (to reduce running time). So this number should be adjusted according to vehicle gear configuration, therefore violates our automated batch processing scheme.
    NUM_TIRES = 6

    # > Vehicle information to be extracted from database
    VEHICLE_FIELDS = ['PercentOfLoad','xCoordinate','yCoordinate','Pressure','ContactArea']

    # > FEM2D output fields
    FEM_FIELDS_2D = ['Displacement_Z', 'Displacement_R', 'Stress_R', 'Stress_Theta', 'Stress_Z', 'Stress_Shear']

    # > FEM3D output fields
    FEM_FIELDS_3D = ['Displacement_X', 'Displacement_Y', 'Displacement_Z', 'Normal_X', 'Normal_Y', 'Normal_Z']

    # > Pavement layer information
    LAYERS = {
        '0' : {'Thickness': 4, 'Modulus': 300000, 'Poisson': 0.35},
        '1' : {'Thickness': 6, 'Modulus': 45000, 'Poisson': 0.35},
        '2' : {'Thickness': 5, 'Modulus': 7500, 'Poisson': 0.35}
    }

    # ================================= #
    # === Manual Input Procedure ====== #
    # ================================= #
    # > Manual input produre or automation read from ERDC database
    MANUAL = True

    MANUAL_TIRE_COORDS = np.array([[0,0]]).reshape(-1,2)

    MANUAL_TIRE_FORCES = np.array([80.0])

    MANUAL_TIRE_AREAS = np.array([246.057])

    MANUAL_TIRE_RADII = np.array([8.85])

    MANUAL_EVAL_POINTS = np.array([[0,0],[8,0],[12,0],[18,0],[24,0],[36,0],[48,0],[60,0],[72,0]]).reshape(-1,2)

    MANUAL_DEPTH_COORDS = np.array([0.0])

    # > Query depth range
    DEPTH = [0, -20]

    # > No. of grid points in Z direction
    DEPTH_POINTS = 20

    DEPTH_COORDS = - (np.logspace(np.log10(-DEPTH[0]+1), np.log10(-DEPTH[1]+1), num=DEPTH_POINTS, base=10) - 1) # logspace
    # DEPTH_COORDS = np.linspace(*self.DEPTH, num=self.DEPTH_POINTS) # linspace
    if MANUAL:
        DEPTH_COORDS = MANUAL_DEPTH_COORDS # user-defined

    # > Analyzed depth for 3D
    DEPTH_3D = [0, -3, -15, -1232]

    # > Mesh density along depth direction
    DEPTH_LINSPACE_3D = 5
    DEPTH_LOGSPACE_3D = 5

    # ================================= #
    # ====== WinJULEA Analysis ======== #
    # ================================= #
    # > Path to save results
    PATH_LEA = './results'

    # > File suffix. e.g., evals_lea.npy, depths_lea.npy
    FILE_SUFFIX_LEA = '_lea'

    # ================================= #
    # === 2D Superposition Analysis ===
    # ================================= #
    # > Path to save results
    PATH_2D = './results'

    # > File suffix. e.g., input_2d.txt, input_2d.vtk, evals_2d.npy
    FILE_SUFFIX_2D = '_2d'

    # > Plot displacement/stress depth profile or not
    # True: plot & save .npy, False: save .npy
    PLOT_2D = False

    # > Superpose full 3D space (True) or just at given evaluation points
    SUPERPOSITION_MESH = False
    # Note: if False, parameters below are not used

    # > Mesh space in terms of p2p distance in X direction
    MESH_RANGE_X_2D = 1

    # > Mesh space in terms of p2p distance in Y direction
    MESH_RANGE_Y_2D = 0.5

    # > Mesh space in terms of tire radius in both X & Y when there is single tire only (e.g. 10xR)
    MESH_RANGE_SINGLE_2D = 10

    # > Densified region around tire location (1.5 means x-1.5R ~ x+1.5R)
    DENSE_REGION_2D = 1.5

    # > Grid spacing in sparse region in terms of radius R
    SPACING_SPARSE_2D = 1/2

    # > Grid spacing in dense region in terms of radius R
    SPACING_DENSE_2D = 1/6

    # ================================= #
    # === 3D Analysis ===
    # ================================= #
    # > Path to save results
    PATH_3D = './results'

    # > File suffix. e.g., input_3d.txt, input_3d.vtk, evals_3d.npy
    FILE_SUFFIX_3D = '_3d'

    # > Plot displacement/stress depth profile or net
    # True: plot & save .npy, False: save .npy
    PLOT_3D = False

    # > Mesh space in terms of p2p distance in X direction
    MESH_RANGE_X_3D = 2

    # > Mesh space in terms of p2p distance in Y direction
    MESH_RANGE_Y_3D = 1.5

    # > Mesh space in terms of tire radius in both X & Y when there is single tire only (e.g. 10xR)
    MESH_RANGE_SINGLE_3D = 10

    # > Densified region around tire location (1.5 means x-1.5R ~ x+1.5R)
    DENSE_REGION_3D = 1.5

    # > Grid spacing in sparse region in terms of radius R
    SPACING_SPARSE_3D = 1

    # > Grid spacing in dense region in terms of radius R
    SPACING_DENSE_3D = 1/4
