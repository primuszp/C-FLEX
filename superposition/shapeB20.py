"""
Class ShapeB20 of the shape functions used for interpolating nodal values when an arbitrary point is queried in FEM mesh.

Copyright (c) 2019 Haohang Huang
Licensed under the GPL License (see LICENSE for details)
Written by Haohang Huang, November 2019.
"""

import numpy as np

class ShapeB20:
    """B20 element shape class for interpolation.
    """
    def __init__(self):
        """Init local coordinates of B20 element nodes.
        """
        nodeCoord = np.zeros((20,3))
        nodeCoord[0,:] = [-1, -1, -1]
        nodeCoord[1,:] = [1, -1, -1]
        nodeCoord[2,:] = [1, 1, -1]
        nodeCoord[3,:] = [-1, 1, -1]
        nodeCoord[4,:] = [-1, -1, 1]
        nodeCoord[5,:] = [1, -1, 1]
        nodeCoord[6,:] = [1, 1, 1]
        nodeCoord[7,:] = [-1, 1, 1]
        nodeCoord[8,:] = [0, -1, -1]
        nodeCoord[9,:] = [1, 0, -1]
        nodeCoord[10,:] = [0, 1, -1]
        nodeCoord[11,:] = [-1, 0, -1]
        nodeCoord[12,:] = [0, -1, 1]
        nodeCoord[13,:] = [1, 0, 1]
        nodeCoord[14,:] = [0, 1, 1]
        nodeCoord[15,:] = [-1, 0, 1]
        nodeCoord[16,:] = [-1, -1, 0]
        nodeCoord[17,:] = [1, -1, 0]
        nodeCoord[18,:] = [1, 1, 0]
        nodeCoord[19,:] = [-1, 1, 0]
        self.nodeCoord = nodeCoord

    def shape_function(self, pts):
        """Compute the shape function vector at query point(s).
        Args:
            pts [n x 3 mat]: Local isoparametric coordinates of query points.
        Returns:
            [n x 20 mat]: shape function vectors (of length num_of_nodes) at each query points.
        Notes:
            See ShapeB20.cpp.
        """
        nodeCoord = self.nodeCoord
        res = np.zeros((len(pts), len(nodeCoord)))
        for i in range(len(nodeCoord)):
            if i < 8: # 4 corner nodes
                res[:,i] = (1 + nodeCoord[i,0] * pts[:,0]) * (1 + nodeCoord[i,1] * pts[:,1]) * (1 + nodeCoord[i,2] * pts[:,2]) * (nodeCoord[i,0] * pts[:,0] + nodeCoord[i,1] * pts[:,1] + nodeCoord[i,2] * pts[:,2] - 2) / 8;
            elif i == 8 or i == 10 or i == 12 or i == 14: # xi = 0 mid-side nodes
                res[:,i] = (1 - pts[:,0] * pts[:,0]) * (1 + nodeCoord[i,1] * pts[:,1]) * (1 + nodeCoord[i,2] * pts[:,2]) / 4
            elif i == 9 or i == 11 or i == 13 or i == 15: # eta = 0 mid-side nodes
                res[:,i] = (1 - pts[:,1] * pts[:,1]) * (1 + nodeCoord[i,0] * pts[:,0]) * (1 + nodeCoord[i,2] * pts[:,2]) / 4
            elif i == 16 or i == 17 or i == 18 or i == 19: # zeta = 0 mid-side nodes
                res[:,i] = (1 - pts[:,2] * pts[:,2]) * (1 + nodeCoord[i,0] * pts[:,0]) * (1 + nodeCoord[i,1] * pts[:,1]) / 4

        return res

    def isoparam_coord(self, coord_nodes, coord_querys):
        """Compute isoparametric coordinate of given point(s) using corner vertices.
        Args:
            coord_nodes [n x 3 mat]: Global coordinates of element nodes.
            coord_querys [n x 3 mat]: Global coordinates of query points.
        Returns:
            [n x 3 mat]: Isoparametric coordinates of queried points.
        """

        # convert global coordinates to local coordinates
        x = (coord_querys[:,0] - coord_nodes[0,0]) / (coord_nodes[1,0] - coord_nodes[0,0])
        y = (coord_querys[:,1] - coord_nodes[0,1]) / (coord_nodes[3,1] - coord_nodes[0,1])
        z = (coord_querys[:,2] - coord_nodes[0,2]) / (coord_nodes[4,2] - coord_nodes[0,2])

        x = self.nodeCoord[0,0] + x * (self.nodeCoord[1,0] - self.nodeCoord[0,0])
        y = self.nodeCoord[0,1] + y * (self.nodeCoord[3,1] - self.nodeCoord[0,1])
        z = self.nodeCoord[0,2] + z * (self.nodeCoord[4,2] - self.nodeCoord[0,2])

        return np.hstack((x.reshape(-1,1), y.reshape(-1,1), z.reshape(-1,1)))

    def query_shape(self, coord_nodes, coord_querys):
        """Compute the shape function(s) at query point(s).
        Args:
            coord_nodes [n x 3 mat]: Global coordinates of element nodes.
            coord_querys [n x 3 mat]: Global coordinates of query points.
        Returns:
            [n x 20 mat]: N shape function vectors (each of length num_of_nodes=20) at N query points.
        """
        return self.shape_function(self.isoparam_coord(coord_nodes, coord_querys))
