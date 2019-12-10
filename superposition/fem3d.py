"""
Run 3D FEM.

Copyright (c) 2019 Haohang Huang
Licensed under the GPL License (see LICENSE for details)
Written by Haohang Huang, November 2019.

Usage: see main.py
"""

import numpy as np
import os
import subprocess
import platform
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from vtk.util.numpy_support import numpy_to_vtk

from .shapeB20 import ShapeB20
from utils.database import Database
from mesh_generator.m3d import MeshGenerator3d
from utils.vtk import readVTK, getCellLocator
from utils.plot import plot_tire_eval, plot_eval_depth
from config import Config as cfg

import warnings
warnings.filterwarnings("ignore")

class FEM3D():
    """3D analysis module.
    Attrs:
        database [Database]: connnected MySQL database obj
        vehicle_name [str]: vehicle name
        tireCoordinates [n x 2]: 2D (x,y) coordinates of each tire
        tireForces [n x 1]: pressure of each tire
        tireAreas [n x 1]: area of each tire
        tireRadii [n x 1]: radius of each tire
        evalPoints [n x 2]: 2D (x,y) coordinates of evaluation points in the database
        elementTypeMap [dict]: VTK element table

        queryMesh [VTK grid]: VTK mesh for Q8 nodal results
        queryPoints [n x 3]: 3D (x,y,z) coordinates of each evaluation points
        depths [n x 1]: depths of the mesh (negative values!)
        interface [L-1 x 1]: interface index in zcoord

        inputFile [str]: input filename
        outputFile [str]: output filename
        results_grid [n x 6]: results at every grid points
        results_eval [n x 6]: results at every eval points
    """
    def __init__(self, database):
        self.database = database
        self.vehicle_name = None
        self.tireCoordinates = None
        self.tireForces = None
        self.tireAreas = None
        self.tireRadii = None
        self.evalPoints = None
        self.elementTypeMap = {
            25 : ShapeB20()
        }

        self.queryMesh = None
        self.queryPoints = None
        self.depths = None
        self.interface = None

        self.inputFile = None
        self.outputFile = None
        self.results_grid = None
        self.results_eval = None

    def run(self, vehicle, num_tires):
        """Main function of FEM3D class. Run the entire analysis of one vehicle.
        Args:
            vehicle [str/int]: vehicle name or ID.
            num_tires [int]: No. of tires to be analyzed (-1 for full gear analysis).
        """
        # 1. Parse vehicle information
        self.vehicle_info(vehicle, num_tires)

        # 2. Generate FEM3D mesh
        self.mesh_generation()

        # 3. Run FEM3D
        self.run_fem3d()

        # 4. Parse result
        self.parse_result()

        # 5. Output
        self.output()

    def vehicle_info(self, vehicle, num_tires):
        """Collect useful vehicle info.
        Args:
            vehicle [str/int]: vehicle name or ID.
            num_tires [int]: No. of tires to be analyzed (-1 for full gear analysis).
        """
        # 0. Get vehicle name
        self.vehicle_name = vehicle if isinstance(vehicle, str) else self.database.vehicle_all()[vehicle-1]

        # 1. Get vehicle's tire configuration and evaluation points
        tires = self.database.query(fields=cfg.VEHICLE_FIELDS, vehicle=self.vehicle_name)

        evals = self.database.query_evaluation(vehicle=self.vehicle_name)

        print("> === Analyzing {} of {} tires of vechicle {} at {} evaluation points ===".format(num_tires, len(tires), self.vehicle_name, len(evals)))

        # 1.1. get full gear configuration by fields
        xy = np.zeros((len(tires), 2))
        force, area = np.zeros(len(tires)), np.zeros(len(tires))
        for i, tire in enumerate(tires):
            _, X, Y, F, A = [tire[f] for f in cfg.VEHICLE_FIELDS]
            xy[i] = [X, Y]
            force[i], area[i] = F, A
        radius = np.sqrt(area / np.pi)

        # 1.2. truncate to a subset of tires when num_tires != -1
        if num_tires != -1:
            xy, force, area, radius = xy[:num_tires,:], force[:num_tires], area[:num_tires], radius[:num_tires]
        else:
            num_tires = len(tires) # update num_tires

        # 1.3. get evaluation points
        pts = np.zeros((len(evals), 2))
        for i, r in enumerate(evals):
            pts[i] = [r['xCoordinate'], r['yCoordinate']]

        self.tireCoordinates = xy
        self.tireForces = force
        self.tireAreas = area
        self.tireRadii = radius
        self.evalPoints = pts

        # 2. print
        print("> Tire Configuration:")
        print("{:8} {:8} {:8} {:8} {:8} {:8}".format("Tire", "X", "Y", "P(psi)", "A(in^2)", "R(in.)"))
        for i in range(num_tires):
            print("{:<8} {:<8.1f} {:<8.1f} {:<8.1f} {:<8.1f} {:<8.1f}".format(i, xy[i,0], xy[i,1], force[i], area[i], radius[i]))
        # print eval points
        print("> Evaluation Points:")
        print("{:8} {:8} {:8} {:8}".format("Point", "X", "Y", "Z"))
        for i, r in enumerate(evals):
            print("{:<8} {:<8.1f} {:<8.1f} {:<8.1f}".format(i, r['xCoordinate'], r['yCoordinate'], r['zCoordinate']))

        # 3. plot
        print("> [Plot] Tire configuration and evaluation points plotted in config.png")
        plot_tire_eval(xy, radius, pts, vehicle=self.vehicle_name, path=os.path.join(cfg.PATH_3D, 'config.png'))

    def mesh_generation(self):
        """Generate 3D FEM mesh.
        """
        # 0.0. collect parameters
        layers = cfg.LAYERS
        depth = cfg.DEPTH_3D
        xy = self.tireCoordinates
        force = self.tireForces
        area = self.tireAreas
        radius = self.tireRadii

        # 0.1. calculate query space and four corner points
        # Note: should handle special cases for single tire or single axle when there is only one unique coordinate along the axis
        num_tires_x, num_tires_y = len(np.unique(xy[:,0])), len(np.unique(xy[:,1]))
        if num_tires_x == 1:
            xlen = np.max(radius) * cfg.MESH_RANGE_SINGLE_3D
        else:
            xlen = xy.ptp(axis=0)[0] * cfg.MESH_RANGE_X_3D
        if num_tires_y == 1:
            ylen = np.max(radius) * cfg.MESH_RANGE_SINGLE_3D
        else:
            ylen = xy.ptp(axis=0)[1] * cfg.MESH_RANGE_Y_3D

        xmin, ymin = xy.min(axis=0)[0], xy.min(axis=0)[1]
        xmax, ymax = xy.max(axis=0)[0], xy.max(axis=0)[1]
        xlim = [xmin - xlen, xmax + xlen]
        ylim = [ymin - ylen, ymax + ylen]
        zlim = depth

        # 1. Generate Q8 rectilinear mesh
        xtire, ytire = np.unique(xy[:,0]), np.unique(xy[:,1]) # unique tire locations in X & Y directions
        rtire = np.max(radius) * cfg.DENSE_REGION_3D # densify within range (x - 1.5R, x + 1.5R) where R is the max radius among all tires
        self._query_mesh(xlim, ylim, zlim, xtire, ytire, rtire)

        # 2. Generate B20 unstructured mesh
        filename = os.path.join(cfg.PATH_3D, 'input'+cfg.FILE_SUFFIX_3D)
        self.inputFile = filename + '.txt'
        self.outputFile = filename + '.vtk'
        xcoords = vtk_to_numpy(self.queryMesh.GetXCoordinates())
        ycoords = vtk_to_numpy(self.queryMesh.GetYCoordinates())
        zcoords = vtk_to_numpy(self.queryMesh.GetZCoordinates())
        mesh_generator = MeshGenerator3d(xcoords, ycoords, zcoords, self.interface)
        mesh_generator.run(self.inputFile, self.tireCoordinates, self.tireForces, self.tireAreas)

    def _nonlinear_spacing(self, space, tire_range, spacing_dense, spacing_sparse):
        """Generate nonlinear spacing (densified around tire location) in 1D coordinates. This function will detect overlapping ranges and merge them into a larger dense region.
        Args:
            space: [min, max] 1D range
            tire_range [n x 2 mat]: ([i,0], [i,1]) left and right range for the tire regions
            spacing_dense [num]: grid spacing for dense range
            spacing_sparse [num]: grid spacing for sparse range
        Returns:
            1D coordinates of nonlinear mesh
        """
        # sparse range (from left boundary to 1st tire range)
        num_segments = np.ceil((tire_range[0,0]-space[0])/spacing_sparse)
        xcoords = np.linspace(space[0], tire_range[0,0], num=num_segments, endpoint=False)
        # (Solved) @BUG: be careful! here we have endpoint=False, and the start point must exist i.e. num should >= 1, so we used ceil() since (tire_range[0,0]-space[0])/spacing_sparse could < 1

        i = 0
        left, right = tire_range[0,0], tire_range[0,1]
        while i < len(tire_range) - 1:
            if right > tire_range[i+1,0]: # overlap detected
                right = tire_range[i+1,1]
                i += 1
                continue
            # dense range
            coord = np.linspace(left, right, num=np.ceil((right-left)/spacing_dense), endpoint=False)
            xcoords = np.concatenate((xcoords, coord))
            # sparse range
            coord = np.linspace(right, tire_range[i+1,0], num=np.ceil((tire_range[i+1,0]-right)/spacing_sparse), endpoint=False)
            xcoords = np.concatenate((xcoords, coord))
            # update
            left, right = tire_range[i+1,0], tire_range[i+1,1]
            i += 1

        # dense range (for last tire range)
        coord = np.linspace(left, right, num=np.ceil((right-left)/spacing_dense), endpoint=False)
        xcoords = np.concatenate((xcoords, coord))

        # sparse range (from last tire to right boundary)
        coord = np.linspace(right, space[1], num=np.ceil((space[1]-right)/spacing_sparse), endpoint=False)

        xcoords = np.concatenate((xcoords, coord, np.array([space[1]]))) # don't forget the right boundary!

        return xcoords

    def _query_mesh(self, xlim, ylim, zlim, xtire, ytire, rtire):
        """Use VTK structured grid to generate query mesh. This is also the mesh of final superposition results.
        Args:
            *lim: [min, max] range for xyz directions
            *tire [n x 1 vec]: unique tire coordinates in x & y directions
            rtire [num]: tire radius to be densified
        Note:
            query mesh will include the grid points & evaluation points at all depth
            In VTK there are 3 types of "structured" grids:
            - vtkImageData (vtkUniformGrid): constant spacing & axis aligned
            - vtkRectilinearGrid: axis aligned & vary spacing
            - vtkStructuredGrid: arbitrarily located points (cells may not be valid).
        """
        # 1. Generate densified XYZ axis coordinates
        spacing_sparse = rtire * cfg.SPACING_SPARSE_3D
        spacing_dense = rtire * cfg.SPACING_DENSE_3D
        xranges = np.hstack([(xtire - rtire).reshape(-1,1), (xtire + rtire).reshape(-1,1)])
        yranges = np.hstack([(ytire - rtire).reshape(-1,1), (ytire + rtire).reshape(-1,1)])
        xcoords = self._nonlinear_spacing(xlim, xranges, spacing_dense, spacing_sparse)
        ycoords = self._nonlinear_spacing(ylim, yranges, spacing_dense, spacing_sparse)
        # Single-layer depth points
        # zcoords = - (np.logspace(np.log10(-zlim[0]+1), np.log10(-zlim[1]+1), num=cfg.DEPTH_POINTS, base=10) - 1) # logspace
        # zcoords = np.linspace(*zlim, num=cfg.DEPTH_POINTS) # linspace

        # Multi-layer depth points
        num_layers = len(zlim) - 1
        zcoord = []
        self.interface = np.zeros(num_layers-1)
        for l in range(num_layers-1):
            zcoord.append(np.linspace(zlim[l], zlim[l+1], num=cfg.DEPTH_LINSPACE_3D, endpoint=False))
            self.interface[l] = cfg.DEPTH_LINSPACE_3D * (l+1)
        zcoord.append(-np.logspace(np.log10(-zlim[-2]), np.log10(-zlim[-1]), num=cfg.DEPTH_LOGSPACE_3D, base=10, endpoint=True))
        zcoords = np.concatenate(zcoord, axis=0)

        # 2. Generate VTK rectilinear grid
        grid = vtk.vtkRectilinearGrid()
        grid.SetDimensions(len(xcoords), len(ycoords), len(zcoords)) # initialize mesh space
        print("> Generating 3D mesh with {} rectilinear points".format(grid.GetNumberOfPoints()))
        grid.SetXCoordinates(numpy_to_vtk(xcoords))
        grid.SetYCoordinates(numpy_to_vtk(ycoords))
        grid.SetZCoordinates(numpy_to_vtk(zcoords))

        coord = vtk.vtkPoints()
        grid.GetPoints(coord)

        self.queryMesh = grid
        # self.queryPoints = vtk_to_numpy(coord.GetData()) # N x 3
        self.queryPoints = np.array([0,0,0]).reshape(-1,3) # don't query point
        # query depths
        self.depths = np.linspace(*cfg.DEPTH, num=cfg.DEPTH_POINTS) # linspace

    def run_fem3d(self):
        """Execute FEM3D program.
        """
        print("> Running FLEX3D for Vehicle {}".format(self.vehicle_name))

        system = platform.system() # find system-specific exe
        subprocess.call(["../bin/{}/main3d".format(system), os.path.join(cfg.PATH_3D, 'input'+cfg.FILE_SUFFIX_3D)])

    def parse_result(self):
        """Parse FEM3D result and get values at grid points and evaluation points.
        """
        print("> Querying results at evaluation points")

        # 1. Read result from VTK
        mesh = readVTK(self.outputFile)
        pointData = mesh.GetPointData()
        fields = np.array([vtk_to_numpy(pointData.GetArray(f)) for f in cfg.FEM_FIELDS_3D]).T # N x 6

        # 2. Find results at grid points (these points must be in mesh, so use FindPoint instead of FindClosestPoint)
        self.results_grid = np.zeros((len(self.queryPoints), len(cfg.FEM_FIELDS_3D)))
        for i in range(len(self.queryPoints)):
            id = mesh.FindPoint(self.queryPoints[i,:])
            self.results_grid[i,:] = fields[id,:]

        # 3. Find results at evaluation points (FindClosestPoint)
        # 3.1. construct evaluation points along all depths
        evals = np.zeros((len(self.evalPoints), len(self.depths), 3)) # P X D X 3
        for i in range(len(self.evalPoints)):
            t = np.tile(self.evalPoints[i], (len(self.depths), 1))
            evals[i] = np.concatenate((t, self.depths.reshape(-1,1)), axis=1)
        evals = evals.reshape(-1, 3)

        # 3.2. search results (similar as 2D)
        eval_result = np.zeros((len(evals), len(cfg.FEM_FIELDS_3D)))
        cellLocator = getCellLocator(mesh)
        closest_point = [0.0, 0.0, 0.0]
        gen_cell = vtk.vtkGenericCell()
        cell_id = vtk.reference(0)
        sub_id = vtk.reference(0)
        dist2 = vtk.reference(0.0)
        X = vtk_to_numpy(pointData.GetArray('Distance_X'))
        Y = vtk_to_numpy(pointData.GetArray('Distance_Y'))
        Z = vtk_to_numpy(pointData.GetArray('Distance_Z'))
        for i in range(len(evals)):
            # find cell
            cellLocator.FindClosestPoint(list(evals[i,:]), closest_point, gen_cell, cell_id, sub_id, dist2)
            elementShape = self.elementTypeMap[mesh.GetCellType(cell_id.get())]

            # get point list of the cell
            vtk_point_list = vtk.vtkIdList()
            mesh.GetCellPoints(cell_id.get(), vtk_point_list)
            point_list = [vtk_point_list.GetId(i) for i in range(vtk_point_list.GetNumberOfIds())]

            # get shape functions at the query point(s)
            coord_nodes = np.hstack((X[point_list].reshape(-1,1), Y[point_list].reshape(-1,1), Z[point_list].reshape(-1,1))) # global coord of element nodes
            coord_querys = evals[i,:].reshape(-1,3) # global coord of query points
            shape_function = elementShape.query_shape(coord_nodes, coord_querys) # see class ShapeQ8

            # interpolate the fields & accumulate/superpose
            interpolated = np.squeeze(np.matmul(shape_function, fields[point_list, :])) # for B20 & 6 fields, 1x20 * 20x6 = 1x6

            eval_result[i,:] = interpolated

        self.results_eval = eval_result.reshape(len(self.evalPoints), cfg.DEPTH_POINTS, len(cfg.FEM_FIELDS_3D)) # P x D x 6

    def output(self):
        """Save evaluation results to npy for plotting.
        """

        if cfg.PLOT_3D:
            plot_eval_depth(self.results_eval, self.depths, cfg.PATH_3D, cfg.FILE_SUFFIX_3D)
            print("> [Plot] Response at evaluation points plotted in eval_x_3d.png")

        print("> Writing evaluation results to NPY")
        np.save(os.path.join(cfg.PATH_3D, 'evals'+cfg.FILE_SUFFIX_3D+'.npy'), self.results_eval)
        np.save(os.path.join(cfg.PATH_3D, 'depths'+cfg.FILE_SUFFIX_3D+'.npy'), self.depths)

if __name__ == '__main__':
    database = Database(name=cfg.DATABASE)

    task = FEM3D(database)
    mode = 'single'
    if mode == 'single':
        # run one vehicle (single mode)
        task.run(cfg.VEHICLE, cfg.NUM_TIRES)
    elif mode == 'batch':
        # run all vehicles (batch mode)
        for vehicle in database.vehicle_all():
            task.run(vehicle, cfg.NUM_TIRES)

    database.close()
