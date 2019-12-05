"""
WinJULEA input/output interface.

Copyright (c) 2019 Jiayi Luo
Licensed under the GPL License (see LICENSE for details)
Written by Jiayi Luo, November 2019.
"""

import os
import sys
import numpy as np

from config import Config as cfg

def generate_input(path):
    """Main function for generating .lea file.
    """
    # thickness, E, poisson ratio, slip
    layer_properties = np.array([[3,400101.103,0.35,0.0], [12,30008.308,0.40,0.0], [-0.1,6004.562,0.45,0.0]]) # 3 layers
    #layer_properties = np.array([[3,400101.103,0.35,0.0], [-0.1,6004.562,0.45,0.0]]) # 1 layer
    # depth_1 = np.linspace(0, -20, 20, endpoint=False)
    # depth_2 = -np.logspace(np.log2(20), np.log2(100.0), num=20, base=2.0)
    # depth = np.concatenate((depth_1, depth_2), axis=0)
    zlim = cfg.DEPTH
    # depth = - (np.logspace(np.log10(-zlim[0]+1), np.log10(-zlim[1]+1), num=cfg.DEPTH_POINTS, base=10) - 1) # logspace
    depth = np.linspace(zlim[0], zlim[1], cfg.DEPTH_POINTS, endpoint=True)

    # X_cor, Y_cor, Pressure, Area
    # Note: winjulea will automatically convert p * A
    loads = np.array([[243.5, 114, 215, 243.247],\
                     [243.5, 57, 215, 243.247],\
                     [243.5, 0, 215, 243.247],\
                     [188.5, 114, 215, 243.247],\
                     [188.5, 57, 215, 243.247],\
                     [188.5, 0, 215, 243.247]])
    loads = np.array([[188.5, 57, 80, 113.097]]) # single tire
    # Evaluation points
    eval_pts = np.array([[216, 57],\
                         [202.2, 57],\
                         [188.5, 57],\
                         [188.5, 28.5]])

    create_lea(path, layer_properties, depth, loads, eval_pts)

def parse_output(path):
    """Main function for parsing .rpt file.
    """
    if not os.path.exists(path):
        sys.exit("Please run WinJULEA and save the report to '{}'".format(path))
    evals, depths = get_plot_input(path, cfg.DEPTH_POINTS, 4)
    return evals, depths


def filter_str(query):
    '''
        Depth: 0, X_cor: 1, Y_cor:2
        Stress_XYZ: 3-5, Disp_XYZ: 15-17
    '''
    query_list = query.split('  ')[1:]
    out = np.array([float(query_list[0]), float(query_list[1]), float(query_list[2]), float(query_list[3]), float(query_list[4]), float(query_list[5]),  float(query_list[15]),  float(query_list[16]), float(query_list[17])])

    return out.reshape(1, -1)

def extract_result(name):
    # Import all the txt results
    with open(name,'r') as f:
        results = f.readlines()

    # Find the staring index
    idx = -1
    for i, l in enumerate(results):
        if l == "*** RESULTS\n":
            idx = i + 4
            break
    idx_max = len(results)

    final = filter_str(results[idx])
    for i in range(idx + 1, idx_max):
        out = filter_str(results[i])
        final = np.concatenate([final, out], axis=0)

    return final

def get_plot_input_(name):
    ret = extract_result(name)
    depth_ = set(ret[:,0])
    depth = [float(d) for d in depth_]
    depth.sort(reverse=True)
    pts_ = set([tuple(pt) for pt in ret[:,1:3]])
    pts = [(float(d[0]), float(d[1])) for d in pts_]

    evals = np.zeros((len(pts),len(depth), 2))
    dic = {}
    for i in range(len(pts)):
        for d in range(len(depth)):
            dic[(pts[i], depth[d])] = [i, d]

    for l in ret:
        [i, d] = dic[((l[1], l[2]), l[0])]
        evals[i, d] = l[-2:]
    return evals, depth

def get_plot_input(name, num_depth, num_eval):
    ret = extract_result(name)
    evals = np.zeros((num_eval, num_depth, 6))
    depths = np.zeros(num_depth)
    for i in range(num_eval):
        for j in range(num_depth):
            m = num_eval*j + i
            if i == 0:
                depths[j] = ret[m, 0]
            evals[i, j, 0] = ret[m, 6]
            evals[i, j, 1] = ret[m, 7]
            evals[i, j, 2] = ret[m, 8]
            evals[i, j, 3] = ret[m, 3]
            evals[i, j, 4] = ret[m, 4]
            evals[i, j, 5] = ret[m, 5]
    return evals, depths

def create_lea(name, layer_properties, depth, loads, eval_pts):
    with open(name, 'w') as f:
        # Write the layer property
        f.write("%d\n" % layer_properties.shape[0])
        for i in range(layer_properties.shape[0]):
            f.write("%.10f\n" % layer_properties[i, 0])
            f.write("%.10f\n" % layer_properties[i, 1])
            f.write("%.10f\n" % layer_properties[i, 2])
            f.write("%.10f\n" % layer_properties[i, 3])

        # Write depth
        f.write("%d\n" % depth.shape[0])
        for i in range(depth.shape[0]):
            f.write("%.10f\n" % depth[i])

        # Write Load
        f.write("%d\n" % loads.shape[0])
        for i in range(loads.shape[0]):
            f.write("%.10f\n" % loads[i, 0])
            f.write("%.10f\n" % loads[i, 1])
            f.write("%.10f\n" % loads[i, 2])
            f.write("%.10f\n" % loads[i, 3])

        # Write eval points
        f.write("%d\n" % eval_pts.shape[0])
        for i in range(eval_pts.shape[0]):
            f.write("%.10f\n" % eval_pts[i, 0])
            f.write("%.10f\n" % eval_pts[i, 1])
