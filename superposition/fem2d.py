"""
Run superposition of 2D FEM.

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

from .shapeQ8 import ShapeQ8
from utils.database import Database
from mesh_generator.m2d import MeshGenerator2d
from utils.vtk import readVTK, getCellLocator
from utils.plot import plot_tire_eval, plot_eval_depth
from config import Config as cfg

import warnings
warnings.filterwarnings("ignore")

class Superposition():
    """2D superposition analysis module.
    Attrs:
        database [Database]: connnected MySQL database obj
        vehicle_name [str]: vehicle name
        tireCoordinates [n x 2]: 2D (x,y) coordinates of each tire
        tireForces [n x 1]: pressure of each tire
        tireAreas [n x 1]: area of each tire
        tireRadii [n x 1]: radius of each tire
        evalPoints [n x 2]: 2D (x,y) coordinates of evaluation points in the database
        elementTypeMap [dict]: VTK element table

        queryMesh [VTK grid]: VTK mesh for final superposition results
        queryPoints [n x 3]: 3D (x,y,z) coordinates of each evaluation points
        depths [n x 1]: depths of the mesh (negative values!)

        inputFileList [list<str>]: input filenames (one for each tire)
        outputFileList [list<str>]: output filenames (one for each tire)
        results_grid [n x 6]: superposition results at every grid points
        results_eval [n x 6]: superposition results at every eval points
    """

    def __init__(self, database):
        print("> === 2D Superposition Module Initialized ===")
        self.database = database
        self.vehicle_name = None
        self.tireCoordinates = None
        self.tireForces = None
        self.tireAreas = None
        self.tireRadii = None
        self.evalPoints = None
        self.elementTypeMap = {
            23 : ShapeQ8()
        }

        self.queryMesh = None
        self.queryPoints = None

        self.inputFileList = None
        self.outputFileList = None
        self.results_grid = None
        self.results_eval = None

    def run(self, vehicle, num_tires):
        """Main function of Superposition class. Run the entire analysis of one vehicle.
        Args:
            vehicle [str/int]: vehicle name or ID.
            num_tires [int]: No. of tires to be analyzed (-1 for full gear analysis).
        """
        # 1. Parse vehicle information
        self.vehicle_info(vehicle, num_tires)

        # 2. Generate FEM2D meshes and superposition mesh (if cfg.SUPERPOSITION_MESH = True)
        self.mesh_generation()

        # 3. Run FEM
        self.rum_fem2d()

        # 4. Superpose
        self.superpose()

        # 5. Output
        self.output()

    def vehicle_info(self, vehicle, num_tires):
        """Collect useful vehicle info.
        Args:
            vehicle [str/int]: vehicle name or ID.
            num_tires [int]: No. of tires to be analyzed (-1 for full gear analysis).
        """
        if cfg.MANUAL:
            print("> === Analyze manual input ===")
            self.tireCoordinates = cfg.MANUAL_TIRE_COORDS
            self.tireForces = cfg.MANUAL_TIRE_FORCES
            self.tireAreas = cfg.MANUAL_TIRE_AREAS
            self.tireRadii = cfg.MANUAL_TIRE_RADII
            self.evalPoints = cfg.MANUAL_EVAL_POINTS
        else:
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
                #xy, force, area, radius = xy[:num_tires,:], force[:num_tires], area[:num_tires], radius[:num_tires]
                xy, force, area, radius = xy[4:5,:], force[4:5], area[4:5], radius[4:5]
                # force[0], area[0], radius[0] = 80, 113.097, 6 # Kim single-tire case
                force[0], area[0], radius[0] = 80, 113.097, 6 # Issam case
                num_tires = 1
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
            plot_tire_eval(xy, radius, pts, vehicle=self.vehicle_name, path=os.path.join(cfg.PATH_2D, 'config.png'))

    def mesh_generation(self):
        """Prepare two types of mesh:
            - FEM2D meshes: one mesh per tire to run FEM2D analysis.
            - Superposition mesh: final output 3D mesh where each grid point is a query point.
        """
        # one tire <--mesh generator 2d--> one input file
        # one input file <--FEM2D--> one output file
        # all output files <--superposition--> one final output file

        # 0.0. collect parameters
        layers = cfg.LAYERS
        xy = self.tireCoordinates
        force = self.tireForces
        area = self.tireAreas
        radius = self.tireRadii

        # 0.1. calculate query space and four corner points
        # Note: should handle special cases for single tire or single axle when there is only one unique coordinate along the axis
        num_tires_x, num_tires_y = len(np.unique(xy[:,0])), len(np.unique(xy[:,1]))
        if num_tires_x == 1:
            xlen = np.max(radius) * cfg.MESH_RANGE_SINGLE_2D
        else:
            xlen = xy.ptp(axis=0)[0] * cfg.MESH_RANGE_X_2D
        if num_tires_y == 1:
            ylen = np.max(radius) * cfg.MESH_RANGE_SINGLE_2D
        else:
            ylen = xy.ptp(axis=0)[1] * cfg.MESH_RANGE_Y_2D

        xmin, ymin = xy.min(axis=0)[0], xy.min(axis=0)[1]
        xmax, ymax = xy.max(axis=0)[0], xy.max(axis=0)[1]
        xlim = [xmin - xlen, xmax + xlen]
        ylim = [ymin - ylen, ymax + ylen]

        # 0.2. calculate radial length needed for FEM2D mesh (based on corners points)
        corners = np.asarray([(x,y) for x in xlim for y in ylim])
        mesh_len = []
        for i in range(len(xy)):
            mesh_len.append(np.max(np.sqrt(np.sum((corners - xy[i])**2, axis=1))))
        mesh_depth = cfg.DEPTH[0]-cfg.DEPTH[1] # not used

        # 1. generate FEM meshes
        self._input_generator(mesh_len, mesh_depth, force, area, layers)

        # 2. generate evaluations points (cross-product xy coord & depth z)
        if len(self.evalPoints) > 0 and len(cfg.DEPTH_COORDS) > 0:
            self.queryPoints = self._evaluation_points(self.evalPoints, cfg.DEPTH_COORDS)

        if cfg.SUPERPOSITION_MESH:
            # 3. Generate query mesh and query grid points
            # finding: query mesh should be much denser around tire coordinate, otherwise when grid spacing is larger than the tire radius, the stress may disappear! (b.c. we never query the loading point with large displacement/stress)
            xtire, ytire = np.unique(xy[:,0]), np.unique(xy[:,1]) # unique tire locations in X & Y directions
            rtire = np.max(radius) * cfg.DENSE_REGION_2D # densify within range (x - 1.5R, x + 1.5R) where R is the max radius among all tires
            gridPoints = self._query_mesh(xlim, ylim, xtire, ytire, rtire)

            # concat evaluation points & mesh grid points
            self.queryPoints = np.concatenate((self.queryPoints, gridPoints), axis=0) # N x 3

    def _input_generator(self, length, depth, force, area, layers):
        """Generate FEM input files in a folder for given vehicle types.
        Args:
            length [list<float>]: radial length of FEM mesh.
            force [list<float>]: tire pressure
            area [list<float>]: tire area
            layers [obj]: layer properties of the pavement (TODO)
        """
        generator = MeshGenerator2d()

        filestem = os.path.join(cfg.PATH_2D, 'input'+cfg.FILE_SUFFIX_2D)
        self.inputFileList = []
        self.outputFileList = []
        for i in range(len(self.tireCoordinates)):
            filename = filestem + '_' + str(i)
            layer_depths = generator.generate_mesh(length[i], depth, force[i], area[i], i, filename+'.txt')
            self.inputFileList.append(filename + '.txt')
            self.outputFileList.append(filename + '.vtk')
        self.layer_depths = layer_depths

    def _evaluation_points(self, xy, z):
        """Generate all evaluation points by cross-product the xy coordinates and depths.
        Args:
            xy [m x 2 mat]: (x,y) coordinates.
            z [n, vec]: z coordinates (negative values).
        Returns:
            [m*n x 3 mat]: reshaped (x,y,z) coordinates of order (m,n,3), i.e. (x1,y1) x all_depths, (x2,y2) x all_depths, etc.
        """
        evalPoints = np.zeros((len(xy), len(z), 3)) # M x N x 3
        for m in range(len(xy)):
            t = np.tile(xy[m], (len(z), 1)) # N x 2
            evalPoints[m] = np.concatenate((t, z.reshape(-1,1)), axis=1)
        return evalPoints.reshape(-1,3)

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

    def _query_mesh(self, xlim, ylim, xtire, ytire, rtire):
        """Use VTK structured grid to generate query mesh. This is also the mesh of final superposition results.
        Args:
            *lim: [min, max] range for xy directions
            *tire [n x 1 vec]: unique tire coordinates in x & y directions
            rtire [num]: tire radius to be densified
        Return:
            [N x 3 mat]: N grid points (x,y,z) of the query mesh.
        Note:
            query mesh will include the grid points & evaluation points at all depth
            In VTK there are 3 types of "structured" grids:
            - vtkImageData (vtkUniformGrid): constant spacing & axis aligned
            - vtkRectilinearGrid: axis aligned & vary spacing
            - vtkStructuredGrid: arbitrarily located points (cells may not be valid).
        """
        # 1. Generate densified XYZ axis coordinates
        spacing_sparse = rtire * cfg.SPACING_SPARSE_2D
        spacing_dense = rtire * cfg.SPACING_DENSE_2D
        xranges = np.hstack([(xtire - rtire).reshape(-1,1), (xtire + rtire).reshape(-1,1)])
        yranges = np.hstack([(ytire - rtire).reshape(-1,1), (ytire + rtire).reshape(-1,1)])
        xcoords = self._nonlinear_spacing(xlim, xranges, spacing_dense, spacing_sparse)
        ycoords = self._nonlinear_spacing(ylim, yranges, spacing_dense, spacing_sparse)
        zcoords = cfg.DEPTH_COORDS

        # 2. Generate VTK rectilinear grid
        grid = vtk.vtkRectilinearGrid()
        grid.SetDimensions(len(xcoords), len(ycoords), len(zcoords)) # initialize mesh space
        print("> Query mesh generated with {} superposition points".format(grid.GetNumberOfPoints()))
        grid.SetXCoordinates(numpy_to_vtk(xcoords))
        grid.SetYCoordinates(numpy_to_vtk(ycoords))
        grid.SetZCoordinates(numpy_to_vtk(zcoords))

        coord = vtk.vtkPoints()
        grid.GetPoints(coord)

        self.queryMesh = grid

        return vtk_to_numpy(coord.GetData()) # N x 3

    def rum_fem2d(self):
        """Execute FEM2D program.
        """
        system = platform.system()
        for i in range(len(self.tireCoordinates)):
            print("> Running C-FLEX2D for Tire {}".format(i))
            subprocess.call(["../bin/{}/main2d".format(system), os.path.join(cfg.PATH_2D, 'input'+cfg.FILE_SUFFIX_2D+'_'+str(i))])

    def superpose(self):
        """Superpose results from multiple output files.
        Returns:
            [N x F mat]: superposition results. N = No. of query points, F = No. of field properties.
        Note:
            self.queryPoints is a concatenation of [evaluation points, query mesh points]. Superposition is done for both together, but results are separated at the end.
        """
        print("> Superposing {} tires".format(len(self.tireCoordinates)))
        superposition = np.zeros((len(self.queryPoints), len(cfg.FEM_FIELDS_3D)))

        # 0. Parameter placeholder for cell locator
        closest_point = [0.0, 0.0, 0.0] # coordinate of closest point (to be returned)
        gen_cell = vtk.vtkGenericCell() # when having many query points, accelerate the cell locator by allocating once
        cell_id = vtk.reference(0) # located cell (to be returned)
        sub_id = vtk.reference(0) # rarely used (to be returned)
        dist2 = vtk.reference(0.0) # squared distance to the closest point (to be returned)

        # 1. Superpose all tires at every query point
        tires = self.tireCoordinates
        for i in range(len(tires)): # for one tire, query all points and accumulate in result
            # mesh data
            mesh = readVTK(self.outputFileList[i])
            pointData = mesh.GetPointData()
            # cellData = mesh.GetCellData()
            cellLocator = getCellLocator(mesh)
            X = vtk_to_numpy(pointData.GetArray('Radial_Distance'))
            Y = vtk_to_numpy(pointData.GetArray('Depth'))
            fields = np.array([vtk_to_numpy(pointData.GetArray(f)) for f in cfg.FEM_FIELDS_2D]).T

            # query all points
            for idx in range(len(self.queryPoints)):
                pt = self.queryPoints[idx]
                radial = np.sqrt((tires[i,0]-pt[0])**2 + (tires[i,1]-pt[1])**2) # radial distance from query point to tire location
                depth = pt[2]

                # find cell
                cellLocator.FindClosestPoint([radial, depth, 0.0], closest_point, gen_cell, cell_id, sub_id, dist2)
                elementShape = self.elementTypeMap[mesh.GetCellType(cell_id.get())]

                # get point list of the cell
                vtk_point_list = vtk.vtkIdList()
                mesh.GetCellPoints(cell_id.get(), vtk_point_list)
                point_list = [vtk_point_list.GetId(i) for i in range(vtk_point_list.GetNumberOfIds())]

                # get shape functions at the query point(s)
                coord_nodes = np.hstack((X[point_list].reshape(-1,1), Y[point_list].reshape(-1,1))) # global coord of element nodes
                coord_querys = np.array([radial, depth]).reshape(-1,2) # global coord of query points
                shape_function = elementShape.query_shape(coord_nodes, coord_querys) # see class ShapeQ8

                # interpolate the fields & accumulate/superpose
                interpolated = np.squeeze(np.matmul(shape_function, fields[point_list, :])) # for Q8 & 6 fields, 1x8 * 8x6 = 1x6
                # now six fields are cfg.FEM_FIELDS_2D = ['Displacement_Z', 'Displacement_R', 'Stress_R', 'Stress_Theta', 'Stress_Z', 'Stress_Shear']

                # decompose radial displacement and stress into plane X-Y
                # projection cos(t) = a * b / (|a||b|) where |a|=|b|=1, normalized
                disp_R, stress_R = interpolated[1], interpolated[2]

                radial_vec = pt[:-1] - tires[i] # vector
                if np.linalg.norm(radial_vec) == 0:
                    # if the query point is exactly AT any tire location, radial_vec is (0,0)
                    disp_X = disp_R
                    disp_Y = disp_R
                    disp_Z = interpolated[0]
                    stress_X = stress_R
                    stress_Y = stress_R
                    stress_Z = interpolated[4]
                else:
                    radial_vec = radial_vec / np.linalg.norm(radial_vec) # normalization
                    disp_X = np.dot(radial_vec, np.array([1,0])) * disp_R
                    disp_Y = np.dot(radial_vec, np.array([0,1])) * disp_R
                    disp_Z = interpolated[0]
                    stress_X = np.dot(radial_vec, np.array([1,0])) * stress_R
                    stress_Y = np.dot(radial_vec, np.array([0,1])) * stress_R
                    stress_Z = interpolated[4]

                # accumulate results
                superposition[idx,:] += np.array([disp_X, disp_Y, disp_Z, stress_X, stress_Y, stress_Z])

        # 2. Separate evaluation point results from mesh points results
        self.results_eval, self.results_grid = np.split(superposition, [len(self.evalPoints) * len(cfg.DEPTH_COORDS)])
        self.results_eval = self.results_eval.reshape(len(self.evalPoints), len(cfg.DEPTH_COORDS), len(cfg.FEM_FIELDS_3D)) # P x D x 6
        for p in range(len(self.results_eval)):
            print("Evaluation point {} ({},{}):".format(p, self.evalPoints[p,0], self.evalPoints[p,1]))
            print(self.results_eval[p][:,2])
        #print("Displacement_Z", self.results_eval[2][:,2])
        #print(self.results_eval[2][0][2]*25.4)

    def output(self):
        """Save superposition results to npy for plotting & Write superposition result to vtk if SUPERPOSITION_MESH = True.
        """
        if cfg.PLOT_2D:
            plot_eval_depth(self.results_eval, cfg.DEPTH_COORDS, cfg.PATH_2D, cfg.FILE_SUFFIX_2D)
            print("> [Plot] Response at evaluation points plotted in eval_x_2d.png")

        print("> Writing evaluation results to NPY")
        np.save(os.path.join(cfg.PATH_2D, 'evals'+cfg.FILE_SUFFIX_2D+'.npy'), self.results_eval)
        np.save(os.path.join(cfg.PATH_2D, 'depths'+cfg.FILE_SUFFIX_2D+'.npy'), cfg.DEPTH_COORDS)

        if cfg.SUPERPOSITION_MESH:
            print("> Writing 2D superposition results to VTK")

            fields = cfg.FEM_FIELDS_3D
            grid = self.queryMesh
            for i, field in enumerate(fields):
                array = numpy_to_vtk(self.results_grid[:, i], array_type=vtk.VTK_DOUBLE)
                array.SetName(field)
                grid.GetPointData().AddArray(array) # add field values

            writer = vtk.vtkRectilinearGridWriter()
            writer.SetInputData(grid)
            writer.SetFileName(os.path.join(cfg.PATH_2D, 'input'+cfg.FILE_SUFFIX_2D+'_final.vtk'))
            writer.Write()
            """Old version
            dataArray = []
            for i in fields:
                array = vtk.vtkDoubleArray()
                array.SetName(i)
                array.SetNumberOfTuples(grid.GetNumberOfPoints()) # axis 0
                array.SetNumberOfComponents(1) # axis 1
                dataArray.append(array)
                grid.GetPointData().AddArray(array)

            for i in range(results.shape[0]): # points
                for j in range(results.shape[1]): # fields
                    dataArray[j].SetValue(i, results[i,j])
            """

if __name__ == "__main__":
    database = Database(name=cfg.DATABASE)

    task = Superposition(database)
    mode = 'single'
    if mode == 'single':
        # run one vehicle (single mode)
        task.run(cfg.VEHICLE, cfg.NUM_TIRES)
    elif mode == 'batch':
        # run all vehicles (batch mode)
        for vehicle in database.vehicle_all():
            task.run(vehicle, cfg.NUM_TIRES)

    database.close()
