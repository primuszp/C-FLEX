import numpy as np
import os
import subprocess
import platform
from pathlib import Path
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from vtk.util.numpy_support import numpy_to_vtk

import sys
sys.path.append('..') # to 'C-FLEX/'

from shape import ShapeQ8
from utils.database import Database
from mesh_generator.m2d import MeshGenerator2d
from utils.vtk import readVTK, getCellLocator
from utils.plot import plot_tire_eval, plot_eval_depth
from config import Config as cfg

import warnings
warnings.filterwarnings("ignore")

class Superposition():
    """Superposition module.
    Attrs:
        database [Database]: connnected MySQL database obj
        vehicle_fields [list<str>]: useful vehicle information in the database
        tireCoordinates [n x 2]: 2D (x,y) coordinates of each tire
        tireForces [n x 1]: pressure of each tire
        tireAreas [n x 1]: area of each tire
        evalPoints [n x 2]: 2D (x,y) coordinates of evaluation points in the database
        depths [n x 1]: depths of the mesh (negative values!)
        queryMesh [VTK grid]: VTK mesh for final superposition results
        queryPoints [n x 3]: 3D (x,y,z) coordinates of each evaluation points
        inputFileList [list<str>]: input filenames (one for each tire)
        outputFileList [list<str>]: output filenames (one for each tire)
        elementTypeMap [dict]: VTK element table
        results [n x 6]: superposition results at every grid points
    """

    def __init__(self, database):
        self.database = database
        self.vehicle_fields = ['PercentOfLoad','xCoordinate','yCoordinate','Pressure','ContactArea']
        self.tireCoordinates = None
        self.tireForces = None
        self.tireAreas = None
        self.evalPoints = None
        self.depths = None
        self.queryMesh = None
        self.queryPoints = None

        self.inputFileList = None
        self.outputFileList = None
        self.fem_fields = ['Displacement_Z', 'Displacement_R', 'Stress_R', 'Stress_Theta', 'Stress_Z', 'Stress_Shear']
        self.fem_fields_3d = ['Displacement_X', 'Displacement_Y', 'Displacement_Z', 'Normal_X', 'Normal_Y', 'Normal_Z']
        self.elementTypeMap = {
            23 : ShapeQ8(),
            25 : ShapeQ8() # TODO
        }

        self.results = None # superpostion results

    def run(self, vehicle, num_tires):
        """Main function of Superposition class. Run the entire analysis of one vehicle.
        Args:
            vehicle [str/int]: vehicle name or ID.
            num_tires [int]: No. of tires to be analyzed (-1 for full gear analysis).
        """
        # === 1. Get vehicle's tire configuration and evaluation points from database === #
        vehicle_name = vehicle if isinstance(vehicle, str) else self.database.vehicle_all()[vehicle-1]

        tires = self.database.query(fields=self.vehicle_fields, vehicle=vehicle)

        evals = self.database.query_evaluation(vehicle=vehicle)

        print("> === Analyzing {} of {} tires of vechicle {}, {} evaluation points ===".format(num_tires, len(tires), vehicle, len(evals)))

        # get full gear configuration
        xy = np.zeros((len(tires), 2))
        force, area = np.zeros(len(tires)), np.zeros(len(tires))
        for i, tire in enumerate(tires):
            _, X, Y, F, A = [tire[f] for f in self.vehicle_fields]
            xy[i] = [X, Y]
            force[i], area[i] = F, A
        radius = np.sqrt(area / np.pi)

        # truncate to a subset of tires when num_tires != -1
        if num_tires != -1:
            xy, force, area, radius = xy[:num_tires,:], force[:num_tires], area[:num_tires], radius[:num_tires]
        else:
            num_tires = len(tires) # update num_tires

        # get evaluation points
        pts = np.zeros((len(evals), 2))
        for i, r in enumerate(evals):
            pts[i] = [r['xCoordinate'], r['yCoordinate']]

        self.tireCoordinates = xy
        self.tireForces = force
        self.tireAreas = area
        self.evalPoints = pts

        # print tire config
        print("> Tire Configuration:")
        print("{:8} {:8} {:8} {:8} {:8} {:8}".format("Tire", "X", "Y", "P(psi)", "A(in^2)", "R(in.)"))
        for i in range(num_tires):
            print("{:<8} {:<8.1f} {:<8.1f} {:<8.1f} {:<8.1f} {:<8.1f}".format(i, xy[i,0], xy[i,1], force[i], area[i], radius[i]))
        # print eval points
        print("> Evaluation Points:")
        print("{:8} {:8} {:8} {:8}".format("Point", "X", "Y", "Z"))
        for i, r in enumerate(evals):
            print("{:<8} {:<8.1f} {:<8.1f} {:<8.1f}".format(i, r['xCoordinate'], r['yCoordinate'], r['zCoordinate']))

        # === 2. Generate query mesh and FEM meshes === #
        # one tire <--mesh generator 2d--> one input file <--FEM2D--> one output file
        layers = cfg.SUPER_LAYERS
        depth = cfg.SUPER_DEPTH

        # calculate query space and corner points
        # Note: handle special cases for single tire or single axle
        num_tires_x, num_tires_y = len(np.unique(xy[:,0])), len(np.unique(xy[:,1]))
        if num_tires_x == 1:
            xlen = np.max(radius) * cfg.SUPER_QUERY_MESH_SINGLE
        else:
            xlen = xy.ptp(axis=0)[0] * cfg.SUPER_QUERY_MESH_X
        if num_tires_y == 1:
            ylen = np.max(radius) * cfg.SUPER_QUERY_MESH_SINGLE
        else:
            ylen = xy.ptp(axis=0)[1] * cfg.SUPER_QUERY_MESH_Y

        xmin, ymin = xy.min(axis=0)[0], xy.min(axis=0)[1]
        xmax, ymax = xy.max(axis=0)[0], xy.max(axis=0)[1]
        xlim = [xmin - xlen, xmax + xlen]
        ylim = [ymin - ylen, ymax + ylen]
        zlim = depth

        # calculate radial length needed for FEM2D mesh (based on corners)
        corners = np.asarray([(x,y) for x in xlim for y in ylim])
        mesh_len = []
        for i in range(num_tires):
            mesh_len.append(np.max(np.sqrt(np.sum((corners - xy[i])**2, axis=1))))

        # 2.1 generate query mesh
        # finding: query mesh should be much denser around tire coordinate, otherwise when grid spacing is larger than the tire radius, the stress may disappear! (b.c. we never query the loading point with large displacement/stress)
        xtire, ytire = np.unique(xy[:,0]), np.unique(xy[:,1]) # unique tire locations in X & Y directions
        rtire = np.max(radius) * cfg.SUPER_DENSE_REGION # densify within range (x - 1.5r, x + 1.5r) where r is the max radius among all tires
        self.query_mesh(xlim, ylim, zlim, xtire, ytire, rtire)

        # 2.2 generate FEM meshes
        self.input_generator(mesh_len, force, area, layers)

        # plot tire configuration and evaluation points (input_generator will clean the folder)
        plot_tire_eval(xy, radius, pts, vehicle=vehicle, path='./results/config.png')
        print("> [Plot] Tire configuration and evaluation points has been plotted in config.png")

        # === 3. Run FEM === #
        self.run_fem()

        # === 4. Run superposition & Write VTK === #
        self.output_superpose()

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

    def query_mesh(self, xlim, ylim, zlim, xtire, ytire, rtire):
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

        spacing_sparse = rtire * cfg.SUPER_SPACING_SPARSE
        spacing_dense = rtire * cfg.SUPER_SPACING_DENSE
        xranges = np.hstack([(xtire - rtire).reshape(-1,1), (xtire + rtire).reshape(-1,1)])
        yranges = np.hstack([(ytire - rtire).reshape(-1,1), (ytire + rtire).reshape(-1,1)])
        xcoords = self._nonlinear_spacing(xlim, xranges, spacing_dense, spacing_sparse)
        ycoords = self._nonlinear_spacing(ylim, yranges, spacing_dense, spacing_sparse)
        zcoords = - (np.logspace(np.log10(-zlim[0]+1), np.log10(-zlim[1]+1), num=cfg.SUPER_DEPTH_POINTS, base=10) - 1) # logspace
        # zcoords = np.linspace(*zlim, num=cfg.SUPER_DEPTH_POINTS) # linspace
        self.depths = zcoords


        grid = vtk.vtkRectilinearGrid()
        grid.SetDimensions(len(xcoords), len(ycoords), len(zcoords)) # initialize mesh space
        print("> Query mesh generated with {} superposition points".format(grid.GetNumberOfPoints()))
        grid.SetXCoordinates(numpy_to_vtk(xcoords))
        grid.SetYCoordinates(numpy_to_vtk(ycoords))
        grid.SetZCoordinates(numpy_to_vtk(zcoords))

        coord = vtk.vtkPoints()
        grid.GetPoints(coord)

        # concat the evaluation points at all depths with queryPoints
        evalPoints = np.zeros((len(self.evalPoints), len(self.depths), 3)) # P X D X 3
        zcoords = zcoords.reshape(-1, 1) # D x 1
        for i in range(len(self.evalPoints)):
            t = np.tile(self.evalPoints[i], (len(self.depths), 1))
            evalPoints[i] = np.concatenate((t, zcoords), axis=1)

        self.queryPoints = np.concatenate((evalPoints.reshape(-1, 3), vtk_to_numpy(coord.GetData())), axis=0) # N x 3
        self.queryMesh = grid

    def input_generator(self, length, force, area, layers):
        """Generate FEM input files in a folder for given vehicle types.
        Args:
            length [list<float>]: radial length of FEM mesh.
            force [list<float>]: tire pressure
            area [list<float>]: tire area
            layers [obj]: layer properties of the pavement (TODO)
        """
        generator = MeshGenerator2d()
        generator.clear_inputs()

        self.inputFileList = []
        for i in range(len(self.tireCoordinates)):
            file = generator.generate_mesh(force[i], area[i], i, length[i], cfg.SUPER_DEPTH[0]-cfg.SUPER_DEPTH[1])
            self.inputFileList.append(file)
        # 'results/input_x.txt'

    def run_fem(self):
        """Execute FEM code.
        """
        system = platform.system()
        self.outputFileList = []
        for i, file in enumerate(self.inputFileList):
            print("> Running C-FLEX2D for Tire {}".format(i))
            subprocess.call(["../bin/{}/main2d".format(system), file])
            filename = Path(file).stem
            self.outputFileList.append('./results/' + filename + ".vtk")

    def output_superpose(self):
        """Superpose results from multiple output files.
        Returns:
            [N x F mat]: superposition results. N = No. of query points, F = No. of field properties.
        Note:
            self.queryPoints is a concatenation of [evaluation points, query mesh points]. Superposition is done for both together, but we should separate them at the end.
        """
        print("> Superposition of {} tires".format(len(self.tireCoordinates)))
        superposition = np.zeros((len(self.queryPoints), len(self.fem_fields_3d)))

        # parameter placeholder for cell locator
        closest_point = [0.0, 0.0, 0.0] # coordinate of closest point (to be returned)
        gen_cell = vtk.vtkGenericCell() # when having many query points, accelerate the cell locator by allocating once
        cell_id = vtk.reference(0) # located cell (to be returned)
        sub_id = vtk.reference(0) # rarely used (to be returned)
        dist2 = vtk.reference(0.0) # squared distance to the closest point (to be returned)

        tires = self.tireCoordinates
        for i in range(len(tires)): # for one tire, query all points and accumulate in result
            # mesh data
            mesh = readVTK(self.outputFileList[i])
            pointData = mesh.GetPointData()
            # cellData = mesh.GetCellData()
            cellLocator = getCellLocator(mesh)
            X = vtk_to_numpy(pointData.GetArray('Radial_Distance'))
            Y = vtk_to_numpy(pointData.GetArray('Depth'))
            fields = np.array([vtk_to_numpy(pointData.GetArray(f)) for f in self.fem_fields]).T

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
                # now six fields are self.fem_fields = ['Displacement_Z', 'Displacement_R', 'Stress_R', 'Stress_Theta', 'Stress_Z', 'Stress_Shear']

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

        # separate evaluation points from query mesh points
        evals, superposition = np.split(superposition, [len(self.evalPoints) * cfg.SUPER_DEPTH_POINTS])
        evals = evals.reshape(len(self.evalPoints), cfg.SUPER_DEPTH_POINTS, 6) # P x D x 6
        # plot
        plot_eval_depth(evals, self.depths, './results/')
        print("> [Plot] Response at evaluation points has been plotted in eval_x.png")
        # save to disk
        np.save('./results/evaluations.npy', evals)

        # write query mesh points result to VTK
        self.results = superposition # cache for validate.py
        self.writeVTK(superposition)

    def writeVTK(self, results):
        """Write superposition result to the query mesh
        """
        print("> Write superposition results to VTK")
        fields = self.fem_fields_3d
        grid = self.queryMesh

        for i, field in enumerate(fields):
            array = numpy_to_vtk(results[:, i], array_type=vtk.VTK_DOUBLE)
            array.SetName(field)
            grid.GetPointData().AddArray(array)

        writer = vtk.vtkRectilinearGridWriter()
        writer.SetInputData(grid)
        writer.SetFileName("./results/superposition.vtk")
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
