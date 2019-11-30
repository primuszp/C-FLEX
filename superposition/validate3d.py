"""
Run 2D FEM superposition and 3D FEM, and validate the results via point-by-point comparison.
"""
import vtk
import numpy as np
import random
import os
from pathlib import Path
from vtk.util.numpy_support import vtk_to_numpy
from vtk.util.numpy_support import numpy_to_vtk

import sys
sys.path.append('..') # to 'C-FLEX/'

from superposition import Superposition
from mesh_generator.m3d import MeshGenerator3d

class Validation():
    def __init__(self, superposition):
        self.superposition = superposition
        self.fem_fields_3d = ['Displacement_X', 'Displacement_Y', 'Displacement_Z', 'Normal_X', 'Normal_Y', 'Normal_Z']

    def run(self, vehicle):
        """Main function of Validation class. Run all analysis for one vehicle.
        """
        # run superposition
        self.superposition.run(vehicle)
        mesh_2d = self.superposition.queryMesh
        result_2d = self.superposition.results

        # generate 3D mesh
        xcoords = vtk_to_numpy(mesh_2d.GetXCoordinates())
        ycoords = vtk_to_numpy(mesh_2d.GetYCoordinates())
        zcoords = vtk_to_numpy(mesh_2d.GetZCoordinates())
        forces = self.superposition.tireForces
        areas = self.superposition.tireAreas
        tireCoordinates = self.superposition.tireCoordinates

        mesh_generator = MeshGenerator3d(xcoords, ycoords, zcoords)
        mesh_generator.run(vehicle+'.txt', tireCoordinates, forces, areas)

        # run FEM3D
        self.run_fem3d(vehicle)

        # read 3D data
        mesh_3d = self.readVTK(vehicle+'.vtk')
        pointData = mesh_3d.GetPointData()
        fields = np.array([vtk_to_numpy(pointData.GetArray(f)) for f in self.fem_fields_3d]).T # N x 6
        queryPoints = self.superposition.queryPoints

        result_3d = np.zeros((len(queryPoints), len(self.fem_fields_3d)))
        for i in range(len(queryPoints)):
            id = mesh_3d.FindPoint(queryPoints[i,:])
            result_3d[i,:] = fields[id,:]

        error = self.result_compare(result_2d, result_3d)

        print("> MAPE statistics:")
        for i in range(len(self.fem_fields_3d)):
            print("> {}: {:.1f}%".format(self.fem_fields_3d[i], error[i]))

    def run_fem3d(self, vehicle):
        """Execute FEM3D code.
        """
        print("> Running FEM3D for vehicle {}".format(vehicle))
        os.system("./main3d \"{}\"".format(vehicle))

    def result_compare(self, result_2d, result_3d):
        print("> Comparing 2D superposition and 3D results")
        error = abs(abs(result_2d) - abs(result_3d))
        return np.mean(error, axis=0) / np.max(abs(result_3d), axis=0) * 100
        # divide by magnitude to avoid divide-by-zero error

    def readVTK(self, filename):
        """Read a VTK file.
        Args:
            filename: ".vtk" file
        Returns:
            VTK mesh data
        """
        reader = vtk.vtkUnstructuredGridReader()
        reader.SetFileName(filename)
        reader.ReadAllVectorsOn()
        reader.ReadAllScalarsOn()
        reader.Update()
        return reader.GetOutput()

if __name__ == '__main__':
    superposition = Superposition()
    task = Validation(superposition)
    mode = 'single'

    vehicles = superposition.link_database(name='erdc')

    if mode == 'single':
        # run one vehicle (single mode)
        #task.run(random.choice(vehicles))
        task.run('Boeing 777-300')
    elif mode == 'batch':
        # run all vehicles (batch mode)
        for vehicle in vehicles:
            task.run(vehicle)

    superposition.close_database()
