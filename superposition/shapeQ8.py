"""
Class ShapeQ8 of the shape functions used for interpolating nodal values when an arbitrary point is queried in FEM mesh.

Copyright (c) 2019 Haohang Huang
Licensed under the GPL License (see LICENSE for details)
Written by Haohang Huang, November 2019.
"""

import numpy as np

class ShapeQ8:
    """Q8 element shape class for interpolation during superpostion.
    """
    def __init__(self):
        """Init local coordinates of Q8 element nodes.
        """
        nodeCoord = np.zeros((8,2))
        nodeCoord[0,:] = [-1,-1]
        nodeCoord[1,:] = [1,-1]
        nodeCoord[2,:] = [1,1]
        nodeCoord[3,:] = [-1,1]
        nodeCoord[4,:] = [0,-1]
        nodeCoord[5,:] = [1,0]
        nodeCoord[6,:] = [0,1]
        nodeCoord[7,:] = [-1,0]
        self.nodeCoord = nodeCoord

    def shape_function(self, pts):
        """Compute the shape function vector at query point(s).
        Args:
            pts [n x 2 mat]: Local isoparametric coordinates of query points.
        Returns:
            [n x 8 mat]: shape function vectors (of length num_of_nodes) at each query points.
        Notes:
            See ShapeQ8.cpp.
        """
        nodeCoord = self.nodeCoord
        res = np.zeros((len(pts), len(nodeCoord)))
        for i in range(len(nodeCoord)):
            if i < 4: # 4 corner nodes
                res[:,i] = (1 + nodeCoord[i,0] * pts[:,0]) * (1 + nodeCoord[i,1] * pts[:,1]) * (nodeCoord[i,0] * pts[:,0] + nodeCoord[i,1] * pts[:,1] - 1) / 4;
            elif i == 4 or i == 6: # xi = 0 mid-side nodes
                res[:,i] = (1 - pts[:,0] * pts[:,0]) * (1 + nodeCoord[i,1] * pts[:,1]) / 2
            elif i == 5 or i == 7: # eta = 0 mid-side nodes
                res[:,i] = (1 - pts[:,1] * pts[:,1]) * (1 + nodeCoord[i,0] * pts[:,0]) / 2

        return res

    def isoparam_coord(self, coord_nodes, coord_querys):
        """Compute isoparametric coordinate of given point(s) using corner vertices.
        Args:
            coord_nodes [n x 2 mat]: Global coordinates of element nodes.
            coord_querys [n x 2 mat]: Global coordinates of query points.
        Returns:
            [n x 2 mat]: Isoparametric coordinates of queried points.
        """

        # convert global coordinates to local coordinates
        x = (coord_querys[:,0] - coord_nodes[0,0]) / (coord_nodes[1,0] - coord_nodes[0,0])
        y = (coord_querys[:,1] - coord_nodes[0,1]) / (coord_nodes[3,1] - coord_nodes[0,1])
        x = self.nodeCoord[0,0] + x * (self.nodeCoord[1,0] - self.nodeCoord[0,0])
        y = self.nodeCoord[0,1] + y * (self.nodeCoord[3,1] - self.nodeCoord[0,1])
        return np.hstack((x.reshape(-1,1), y.reshape(-1,1)))

    def query_shape(self, coord_nodes, coord_querys):
        """Compute the shape function(s) at query point(s).
        Args:
            coord_nodes [n x 2 mat]: Global coordinates of element nodes.
            coord_querys [n x 2 mat]: Global coordinates of query points.
        Returns:
            [n x 8 mat]: N shape function vectors (each of length num_of_nodes) at N query points.
        """
        return self.shape_function(self.isoparam_coord(coord_nodes, coord_querys))
