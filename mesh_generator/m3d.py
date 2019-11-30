# -*- coding: utf-8 -*-
import vtk
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

class MeshGenerator3d:

    def refine_coords(self, x):
        out_x = []
        for i in range(len(x) - 1):
            out_x.append(x[i])
            out_x.append((x[i] + x[i + 1])/2)
        out_x.append(x[-1])
        return np.array(out_x)

    def getCellId(self, coords, dim):
        '''
            Input: Global Cell Coordinates
            Return: Global Cell Id
        '''
        return (dim[0]*dim[1])*coords[2] + dim[0]*coords[1] + coords[0]

    def __init__(self, x, y, h):

        # origins are the target mesh coordinates
        self.x_origin = x
        self.y_origin = y
        self.h_origin = h

        # change the refined x, y, h mesh
        self.x = self.refine_coords(self.x_origin)
        self.y = self.refine_coords(self.y_origin)
        self.h = self.refine_coords(self.h_origin)
        self.dim = [len(self.x) - 1, len(self.y) - 1, len(self.h) - 1]

        # Layer information
        self.E = []
        self.layer_thickness = h[0] - h[-1]
        B_F = (0,0) # Usually can be (0, -0.0807)
        self.E.append([300000.0, 0.35, B_F[0], B_F[1], 6.5*10**(-6), 0])

        self.analysis = []
        self.analysis.append([0, 0, 0])# analysis[0]->anisotropic; analysis[1]->nonlinear; analysis[2]->no-tension; analysis[3]->viscoelastic or not

        # Extracting indices for each element
        # Each Cell is composed by eight smaller cells
        '''
        The preliminary indices config for type 25 (20-noded) element is:

            xy-plane 1st level:
                7 - 8 - 10
                |        |
                2        5
                |        |
                0 - 16 - 4

            xy-plane 2nd level:
                9 - - - 11
                |        |
                |        |
                |        |
                3 - -  - 6

            xy-plane 3rd level:
                17 - 18 - 19
                |         |
                14        16
                |         |
                12 - 13 - 15

        The target indices config (in vtk and ParaView):

            xy-plane 1st level:
                0 -- 11 -- 3
                |          |
                8         10
                |          |
                1 -- 9 --  2

            xy-plane 2nd level:
                16 ------- 19
                |          |
                |          |
                |          |
                17 ------- 18

            xy-plave 3rd level:
                4 -- 15 -- 7
                |          |
                12         14
                |          |
                5 -- 13 -- 6
        '''

        # Transform indices list: map from each local coordinates (small element) to the Big Cell's
        idx_0 = [0, 1, 2, 4]
        idx_1 = [1, 3, 5]
        idx_2 = [2, 3, 6]
        idx_3 = [3, 7]
        idx_4 = [4, 5, 6]
        idx_5 = [5, 7]
        idx_6 = [6, 7]
        idx_7 = [7]
        self.idx_list = [idx_0, idx_1, idx_2, idx_3, idx_4, idx_5, idx_6, idx_7]

        # Permute indices list: map from the preliminary coordinates to the global coordinates
        self.permute_idx = [0,4,10,7,12,15,19,17,1,5,8,2,13,16,18,14,3,6,11,9]

    def add_load(self, centers, forces, areas):
        self.forces = forces
        self.areas = areas
        self.centers = centers
        self.radius = np.sqrt(areas/np.pi)

        # calculate the valid range
        x_min = centers[:, 0] - self.radius
        x_max = centers[:, 0] + self.radius
        y_min = centers[:, 1] - self.radius
        y_max = centers[:, 1] + self.radius

        # Calculate the center
        center_x = (self.x_origin[0:-1] + self.x_origin[1:])/2
        center_y = (self.y_origin[0:-1] + self.y_origin[1:])/2
        num_forces = forces.shape[0]
        cell_lists = []
        for i in range(num_forces):
            cell_list = []
            idx_x = set(np.where(center_x >= x_min[i])[0]) & set(np.where(center_x <= x_max[i])[0])
            idx_y = set(np.where(center_y >= y_min[i])[0]) & set(np.where(center_y <= y_max[i])[0])
#            idx = list(idx_x & idx_y)
            for j in idx_x:
                for m in idx_y:
                    if (centers[i, 0] - center_x[j])**2 + (centers[i, 1] - center_y[m])**2 <= self.radius[i]**2:
                        cell_list.append([j,m])
            cell_lists.append(cell_list)
        self.cell_lists = cell_lists
        self.center_x = center_x
        self.center_y = center_y

    def apply_load(self):
        uGrid = self.uGrid
        cell_locator = vtk.vtkCellLocator()
        cell_locator.SetDataSet(uGrid)
        cell_locator.BuildLocator()
        closest_point = [0.0, 0.0, 0.0] # coordinate of closest point (to be returned)
        gen_cell = vtk.vtkGenericCell() # when having many query points, accelerate the cell locator by allocating once
        cell_id = vtk.reference(0) # located cell (to be returned)
        sub_id = vtk.reference(0) # rarely used (to be returned)
        dist2 = vtk.reference(0.0)

        self.applied_forces = []
        for i,l in enumerate(self.cell_lists):
            force = self.forces[i]
            for ids in l:
                cell_locator.FindClosestPoint([self.center_x[ids[0]], self.center_y[ids[1]], 0.0], closest_point, gen_cell, cell_id, sub_id, dist2)
                cellId = cell_id.get()
                force_info = [cellId, force, 0]
                self.applied_forces.append(force_info)

    def construct_mesh(self):

        # Generate Rectangular Mesh
        grid = vtk.vtkRectilinearGrid()
        grid.SetDimensions(len(self.x), len(self.y), len(self.h)) # initialize mesh space
        grid.SetXCoordinates(numpy_to_vtk(self.x))
        grid.SetYCoordinates(numpy_to_vtk(self.y))
        grid.SetZCoordinates(numpy_to_vtk(self.h))
        self.grid = grid

        # Insert points
        coord = vtk.vtkPoints()
        self.grid.GetPoints(coord)
        self.pts = vtk_to_numpy(coord.GetData())
        points = vtk.vtkPoints()
        points.Initialize()
        for pt in self.pts:
            points.InsertNextPoint(pt[0], pt[1], pt[2])
        self.points = points

        # Generate unStructured Mesh
        uGrid = vtk.vtkUnstructuredGrid()
        uGrid.SetPoints(self.points)
        self.cell_pt_list = []
        for i in range(len(self.x_origin) - 1):
            for j in range(len(self.y_origin) - 1):
                for k in range(len(self.h_origin) - 1):
                    cell_id_list = []

                    # 0,0,0
                    coords = [2*i, 2*j, 2*k]
                    cell_id = self.getCellId(coords, self.dim)
                    cell = grid.GetCell(cell_id)
                    output_ids = np.array([cell.GetPointIds().GetId(m) for m in range(cell.GetNumberOfPoints())])
                    cell_id_list+=(list(output_ids[self.idx_list[0]]))

                    # 1,0,0
                    coords = [2*i + 1, 2*j, 2*k]
                    cell_id = self.getCellId(coords, self.dim)
                    cell = grid.GetCell(cell_id)
                    output_ids = np.array([cell.GetPointIds().GetId(m) for m in range(cell.GetNumberOfPoints())])
                    cell_id_list+=(list(output_ids[self.idx_list[1]]))

                    # 0,1,0
                    coords = [2*i, 2*j + 1, 2*k]
                    cell_id = self.getCellId(coords, self.dim)
                    cell = grid.GetCell(cell_id)
                    output_ids = np.array([cell.GetPointIds().GetId(m) for m in range(cell.GetNumberOfPoints())])
                    cell_id_list+=(list(output_ids[self.idx_list[2]]))

                    # 1,1,0
                    coords = [2*i + 1, 2*j + 1, 2*k]
                    cell_id = self.getCellId(coords, self.dim)
                    cell = grid.GetCell(cell_id)
                    output_ids = np.array([cell.GetPointIds().GetId(m) for m in range(cell.GetNumberOfPoints())])
                    cell_id_list+=(list(output_ids[self.idx_list[3]]))

                    # 0,0,1
                    coords = [2*i, 2*j, 2*k + 1]
                    cell_id = self.getCellId(coords, self.dim)
                    cell = grid.GetCell(cell_id)
                    output_ids = np.array([cell.GetPointIds().GetId(m) for m in range(cell.GetNumberOfPoints())])
                    cell_id_list+=(list(output_ids[self.idx_list[4]]))

                    # 1,0,1
                    coords = [2*i + 1, 2*j, 2*k + 1]
                    cell_id = self.getCellId(coords, self.dim)
                    cell = grid.GetCell(cell_id)
                    output_ids = np.array([cell.GetPointIds().GetId(m) for m in range(cell.GetNumberOfPoints())])
                    cell_id_list+=(list(output_ids[self.idx_list[5]]))

                    # 0,1,1
                    coords = [2*i, 2*j + 1, 2*k + 1]
                    cell_id = self.getCellId(coords, self.dim)
                    cell = grid.GetCell(cell_id)
                    output_ids = np.array([cell.GetPointIds().GetId(m) for m in range(cell.GetNumberOfPoints())])
                    cell_id_list+=(list(output_ids[self.idx_list[6]]))

                    # 1,1,1
                    coords = [2*i + 1, 2*j + 1, 2*k + 1]
                    cell_id = self.getCellId(coords, self.dim)
                    cell = grid.GetCell(cell_id)
                    output_ids = np.array([cell.GetPointIds().GetId(m) for m in range(cell.GetNumberOfPoints())])
                    cell_id_list+=(list(output_ids[self.idx_list[7]]))

                    final_cell_id_list = np.array(cell_id_list)[self.permute_idx]
                    self.cell_pt_list.append(final_cell_id_list)

                    # Insert into uGrid
                    hex_ = vtk.vtkQuadraticHexahedron()
                    for m in range(0, 20):
                        hex_.GetPointIds().SetId(m, final_cell_id_list[m])
                    uGrid.InsertNextCell(hex_.GetCellType(), hex_.GetPointIds())

        self.uGrid = uGrid

    def setX(self, x):
        self.x_origin = x

    def setY(self, y):
        self.y_origin = y

    def setZ(self, h):
        self.h_origin = h

    def fixBoundary(self):
        pts = self.true_pts
        left_boundary_list = np.where(pts[:,0] == self.x[0])
        right_boundary_list = np.where(pts[:,0] == self.x[-1])
        front_boundary_list = np.where(pts[:,1] == self.y[0])
        back_boundary_list = np.where(pts[:,1] == self.y[-1])
        bottom_boundary_list = np.where(pts[:,2] == self.h[-1])

        # x-direction fixed pt list
        self.x_fixed = list(left_boundary_list[0]) + list(right_boundary_list[0]) + \
                  list(front_boundary_list[0]) + list(back_boundary_list[0]) + \
                  list(bottom_boundary_list[0])

        # y-direction fixed pt list
        self.y_fixed = list(left_boundary_list[0]) + list(right_boundary_list[0]) + \
                  list(front_boundary_list[0]) + list(back_boundary_list[0]) + \
                  list(bottom_boundary_list[0])

        # h-direction fixed pt list
        self.h_fixed = list(bottom_boundary_list[0])

    def set_pts(self):
        out = []
        for l in self.cell_pt_list:
            out = out + list(l)
        trans = list(set(out))
        self.transform_indices = np.array(trans)
        self.true_pts = self.pts[trans]

    def get_true_id(self, idx):
        return np.where(self.transform_indices == idx)[0][0]

    # For test purpose
    def write2VTK(self, path):
        writer = vtk.vtkUnstructuredGridWriter()
        writer.SetInputData(self.uGrid)
        writer.SetFileName(path)
        writer.Write()

    def write2TXT(self, name):
        '''
            Output as an input.txt file
        '''

        # Output file name
        f = open(name,"w")
        num_elem = self.uGrid.GetNumberOfCells()
        num_node = self.true_pts.shape[0]
        num_layer = 1
        # Output essential keys
        # (number of nodes, number of layers, number of x-direction point load, y-direction point load...
        #  edge load, x_prescribed disp., y_prescribed disp.)
        f.write("%d %d %d %d %d %d %d %d %d\r\n" %\
                (num_node, num_elem, num_layer, 0, 0, len(self.applied_forces), len(self.x_fixed), len(self.y_fixed), len(self.h_fixed)))

        # Output layer information
        for i in range(num_layer):
            #f.write("%d %d %d %d %d"%(elem_num[i], elem_num[i+1]-1, analysis[i][0], analysis[i][1], analysis[i][2]))
            f.write("%d %d %d %d %d"%(0, num_elem - 1, self.analysis[i][0], self.analysis[i][1], self.analysis[i][2]))
            f.write("\r\n")
            for j in range(len(self.E[i])):
                f.write("%f"%(self.E[i][j]))
                f.write("%s"%" ")
            f.write("\r\n")
        '''
        # Output model information
        f.write("%d\r\n"%(model_num))
        for i in range(len(model_para)):
            f.write("%.4f "%(model_para[i]))
        f.write("\r\n")
        for i in range(len(iter_para)):
            f.write("%.4f "%(iter_para[i]))
        f.write("\r\n")

        # Output y-direction edge load
        for i in y_force_edge:
            f.write("%d %d\r\n"%(i,2))
            f.write("%.4f %.4f\r\n"%(0,y_force_edge[i]))
        '''
        # Output x-direction edge load
        for l in self.applied_forces:
            f.write("%d %d\r\n"%(l[0], l[2]))
            f.write("%.4f %.4f %.4f\r\n"%(0,0,-l[1]))

        # Output node information
        for i in range(num_node):
            f.write("%.4f %.4f %.4f\r\n" %(self.true_pts[i][0], self.true_pts[i][1], self.true_pts[i][2]))

        for l in self.cell_pt_list:
            print(20, end=' ', file=f)
            for pt_id in l:
                print(self.get_true_id(pt_id), end=' ', file=f)
            print('', file=f)

        # Output element information
#        for i in range(num_elem):
#            f.write("%d %d %d %d %d %d %d %d %d\r\n" %\
#                (20,self.cell_pt_list[i],elem[i][1],elem[i][2],elem[i][3],elem[i][4],elem[i][5],elem[i][6],elem[i][7]))

        # Output x-direction prescribed disp.
        for i in self.x_fixed:
            f.write("%d "%i)
        if (len(self.x_fixed)!=0):
            f.write("\r\n")
        for i in self.x_fixed:
            f.write("%.5f "%0)
        if (len(self.x_fixed)!=0):
            f.write("\r\n")

        # Output y-direction prescribed disp.
        for i in self.y_fixed:
            f.write("%d "%i)
        if (len(self.y_fixed)!=0):
            f.write("\r\n")
        for i in self.y_fixed:
            f.write("%.5f "%0)
        if (len(self.x_fixed)!=0):
            f.write("\r\n")

        # Output h-direction prescribed disp.
        for i in self.h_fixed:
            f.write("%d "%i)
        if (len(self.h_fixed)!=0):
            f.write("\r\n")
        for i in self.h_fixed:
            f.write("%.5f "%0)
        f.close()
    def run(self, txt_path, centers, forces, areas):
        self.add_load(centers, forces, areas)
        self.construct_mesh()
        self.set_pts()
        self.fixBoundary()
        self.apply_load()
        self.write2TXT(txt_path)

if __name__ == '__main__':
    # === Input ==== #
    x = np.array([0, 6, 12, 18, 24, 30, 36])
    y = np.array([0, 6, 12, 18, 24, 30, 36])
    z = np.array([0, -6, -12, -18])
    centers = np.array([[12, 12], [24, 24]])
    forces = np.array([10, 10])
    areas = np.array([400, 400])
    m = meshGenerator3d(x, y, z)
    m.run("vehicle.txt", centers, forces, areas)
