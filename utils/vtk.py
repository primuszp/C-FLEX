"""
VTK utilities.

Copyright (c) 2019 Haohang Huang
Licensed under the GPL License (see LICENSE for details)
Written by Haohang Huang, November 2019.
"""

import vtk

def readVTK(filename):
    """Read a VTK file into mesh.
    Args:
        filename [str]: ".vtk" file
    Returns:
        VTK mesh object
    """
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(filename)
    reader.ReadAllVectorsOn()
    reader.ReadAllScalarsOn()
    reader.Update()
    return reader.GetOutput()

def getCellLocator(mesh):
    """VTK cell locator (find the cell where a query point lies)
    Args:
        mesh: VTK grid object
    Returns:
        VTK cell locator
    Doc:
        https://vtk.org/doc/nightly/html/classvtkCellLocator.html#aeeebba8f210df3851cb50315f9f49b72
    Tutorial:
        https://stackoverflow.com/questions/50468069/vtkcelllocator-findclosestpoint-usage-in-python
        https://lorensen.github.io/VTKExamples/site/Cxx/PolyData/CellLocator/
    """
    cell_locator = vtk.vtkCellLocator()
    cell_locator.SetDataSet(mesh)
    cell_locator.BuildLocator()
    return cell_locator
