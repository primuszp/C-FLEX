"""
Main entrance of superposition module, including WinJULEA run, FEM2D superposition run, FEM3D run, and comparison.

Copyright (c) 2019 Haohang Huang
Licensed under the GPL License (see LICENSE for details)
Written by Haohang Huang, November 2019.
"""

import os
import numpy as np
import sys
sys.path.append('..') # to 'C-FLEX/'

from config import Config as cfg
from utils.database import Database
from utils.plot import *
from superposition.winjulea import generate_input as winjulea_in
from superposition.winjulea import parse_output as winjulea_out
from superposition.fem2d import Superposition
from superposition.fem3d import FEM3D

# control panel
CLEAN = 0
LEA_IN = 0
LEA_OUT = 0
D2 = 1
D3 = 0
PLOT = 0

def main():
    # clean output folder
    if CLEAN:
        clean_folder(cfg.PATH_2D)
        clean_folder(cfg.PATH_3D)

    database = Database(name=cfg.DATABASE)

    # LEA
    if LEA_IN:
        winjulea_in(os.path.join(cfg.PATH_LEA, 'winjulea.lea'))
    if LEA_OUT:
        evals, depths = winjulea_out(os.path.join(cfg.PATH_LEA, 'winjulea.rpt'))
        np.save(os.path.join(cfg.PATH_LEA, 'evals'+cfg.FILE_SUFFIX_LEA+'.npy'), evals)
        np.save(os.path.join(cfg.PATH_LEA, 'depths'+cfg.FILE_SUFFIX_LEA+'.npy'), depths)

    # 2D
    if D2:
        d2 = Superposition(database)
        d2.run(cfg.VEHICLE, cfg.NUM_TIRES)

    # 3D
    if D3:
        d3 = FEM3D(database)
        d3.run(cfg.VEHICLE, cfg.NUM_TIRES)

    # plot
    if PLOT:
        """
        # LEA vs superposition 2D
        evals_lea = np.load(os.path.join(cfg.PATH_LEA, 'evals'+cfg.FILE_SUFFIX_LEA+'.npy'))
        depths_lea = np.load(os.path.join(cfg.PATH_LEA, 'depths'+cfg.FILE_SUFFIX_LEA+'.npy'))
        evals_2d = np.load(os.path.join(cfg.PATH_2D, 'evals'+cfg.FILE_SUFFIX_2D+'.npy'))
        depths_2d = np.load(os.path.join(cfg.PATH_2D, 'depths'+cfg.FILE_SUFFIX_2D+'.npy'))
        plot_eval_depth_lea_2d(evals_lea, depths_lea, evals_2d, depths_2d, cfg.PATH_LEA)
        """

        # LEA vs superposition 2D vs 3D
        evals_lea = np.load(os.path.join(cfg.PATH_LEA, 'evals'+cfg.FILE_SUFFIX_LEA+'.npy'))
        depths_lea = np.load(os.path.join(cfg.PATH_LEA, 'depths'+cfg.FILE_SUFFIX_LEA+'.npy'))
        evals_2d = np.load(os.path.join(cfg.PATH_2D, 'evals'+cfg.FILE_SUFFIX_2D+'.npy'))
        depths_2d = np.load(os.path.join(cfg.PATH_2D, 'depths'+cfg.FILE_SUFFIX_2D+'.npy'))
        evals_3d = np.load(os.path.join(cfg.PATH_3D, 'evals'+cfg.FILE_SUFFIX_3D+'.npy'))
        depths_3d = np.load(os.path.join(cfg.PATH_3D, 'depths'+cfg.FILE_SUFFIX_3D+'.npy'))
        plot_eval_depth_all(evals_lea, depths_lea, evals_2d, depths_2d, evals_3d, depths_3d, cfg.PATH_3D)

    database.close()

def clean_folder(name):
    for f in os.listdir(name):
        os.unlink(os.path.join(name, f))

if __name__ == '__main__':
    main()
