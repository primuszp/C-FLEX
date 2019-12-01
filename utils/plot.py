"""
Plot functions.

Copyright (c) 2019 Haohang Huang
Licensed under the GPL License (see LICENSE for details)
Written by Haohang Huang, November 2019.
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_tire_eval(xy, radius, pts, vehicle=None, path=None):
    """Plot tire configuration and evaluation points.
    Args:
        xy [n x 2]: tire locations.
        radiums [n x 1]: tire radii.
        pts [n x 2]: evaluation point locations.
        vehicle [str]: vehicle name.
        path [str]: path for saving figure. None if plot only.
    """
    xmin, ymin = xy.min(axis=0)[0], xy.min(axis=0)[1]
    xmax, ymax = xy.max(axis=0)[0], xy.max(axis=0)[1]
    xlen, ylen = xy.ptp(axis=0)[0], xy.ptp(axis=0)[1]
    xlim = [xmin - xlen, xmax + xlen]
    ylim = [ymin - ylen/2, ymax + ylen/2]

    fig = plt.figure()
    ax = fig.subplots()
    plt.xlim(xlim)
    plt.ylim(ylim)
    ax.set_axisbelow(True)
    ax.minorticks_on()
    plt.grid(which='major', linestyle='-', color='gray')
    plt.grid(which='minor', linestyle='--')
    ax.set_aspect(1)

    # plot tires
    for i in range(len(xy)):
        ax.add_artist(plt.Circle((xy[i,0], xy[i,1]), radius[i], color='b'))
        ax.add_artist(plt.Text(xy[i,0], xy[i,1], str(i), color='white', va='center', ha='center'))

    # plot eval points
    ax.scatter(pts[:,0], pts[:,1], marker='*', color='red', zorder=5)
    for i in range(len(pts)):
        ax.add_artist(plt.Text(pts[i,0], pts[i,1]-2, str(i), color='black', va='top', ha='center'))

    # legend
    custom_legend = [
        plt.Line2D([],[], marker='o', color='blue', linestyle='None'),
        plt.Line2D([],[], marker='*', color='red', linestyle='None')
    ]
    ax.legend(custom_legend, ['Tire Location', 'Evaluation Point'], loc='upper right', framealpha=1)

    if vehicle != None:
        plt.title("Configuration of {}".format(vehicle))
    plt.xlabel("X Coordinate (in.)")
    plt.ylabel("Y Coordinate (in.)")

    if path != None:
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def plot_eval_depth(evals, depths, path=None):
    """Plot results at evaluation points along depth.
    Args:
        evals [P x D x 6]: superposition results at evaluation points. P is No. of points, D is No. of depth, 6 is result fields ['Displacement_X', 'Displacement_Y', 'Displacement_Z', 'Normal_X', 'Normal_Y', 'Normal_Z'].
        depths [D x 1]: depth values (negative!)
        path [str]: path for saving figure. None if plot only.
    """
    # legend
    custom_legend = [
        plt.Line2D([],[], marker='o', color='red', linestyle='None'),
        plt.Line2D([],[], marker='o', color='blue', linestyle='None'),
        plt.Line2D([],[], marker='o', color='black', linestyle='None')
    ]

    for i in range(len(evals)):
        fig = plt.figure()
        ax = fig.subplots(ncols=2) # y is depth
        plt.suptitle('Responses at Evaluation Point {}'.format(i), y=0.01)

        data = evals[i] # D x 6
        ax[0].plot(data[:,2], depths, marker='o', color='blue')
        ax[0].set_xlabel('Vertical Displacement (in.)')
        ax[1].plot(data[:,5], depths, marker='o', color='blue')
        ax[1].set_xlabel('Vertical Stress (psi)')

        for j in range(2): # two subplots
            ax[j].xaxis.set_ticks_position('top')
            ax[j].xaxis.set_label_position('top')
            ax[j].minorticks_on()
            ax[j].grid(which='major', linestyle='-', color='gray')
            ax[j].grid(which='minor', linestyle='--')
            ax[j].set_ylabel('Depth (in.)')
            ax[j].tick_params(labelsize='small')
            ax[j].legend(custom_legend, ['WinJULEA', '2D Superposition', '3D'], loc='lower right', fontsize='small', framealpha=0.5)

        plt.tight_layout()

        if path != None:
            plt.savefig(path+'eval_{}.png'.format(i), dpi=300, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()

if __name__ == '__main__':
    import random
    evals = np.random.rand(1, 10, 6)
    depths = np.linspace(0,-10, num=10)
    plot_eval_depth(evals, depths)
