#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sun Apr 22 03:52:19 2018

@author: luojiayi

Mesh Generator: Automatic generate mesh with simple inputs. All the values that need to be
                modified are located between two #=====...=====# lines. All the rest part
                should not be changed.

Problem so far:
    1) Edge load direction (x-direction?  Now using 3).
    2) Several different Python file for different input type
    3) Need to change the way to add body force
"""

# Import Required Functions
import numpy as np
import os

class MeshGenerator2d:
    '''
        Generate mesh
    '''
    def __init__(self):
        self.done = False

    def get_force(self):
        return self.force

    def get_radius(self):
        return self.radius

    def clear_inputs(self):
        for f in os.listdir('./results/'):
            os.unlink(os.path.join('./results', f))

    def generate_mesh(self, force, area, num, x_len = -1, y_len=-1):
        radius = np.sqrt(area/np.pi)
        num_layer = 1
        layer_thickness = np.array([y_len])
        # layer_thickness = np.array([4, 4, 6, 4, 222])
        # layer_thickness = np.array([150 * 1 / 25.4])
        # ========================================================================================#

        model_num = 2
        model_para = [2212.5, 0.6577, -0.0657]
        iter_para = [0, 0, 0.3, 0.3] # 5: gravity incremental #; loading incremental #; damping ratio for each increment

        # Apply the properties for each layer
        E = []
        # ========================================================================================#
        '''
            Required Inputs for each layer information:
            Args:
                @ E(list): A list of all the layers and the properties of dimension (num_layer)
                @ B_F(tuple): A tuple for the body force. B_F[0]->direction, 0->down; B_F[1]->amplitude
        '''
        B_F = (0,0) # Usually can be (0, -0.0807)
        E.append([300000.0, 0.35, B_F[0], B_F[1], 6.5*10**(-6), 0])
        #E.append([45000.0, 0.30, B_F[0], B_F[1], 6.5*10**(-6), 0])
        #E.append([7500.0, 0.45, B_F[0], B_F[1], 6.5*10**(-6), 0])
        #E.append([6750,45000,0.15,0.45,15734,0, -0.0856, 6.5*10**(-6), 0])
        #E.append([7500.0, 0.4, B_F[0], B_F[1], 6.5*10**(-6), 0])# Do not modify two B_F terms
        # E.append([20000.0, 0.2, B_F[0], -0.0856, 6.5*10**(-6), 0])
        # E.append([6750,45000,0.15,0.45,15734,0, -0.0856, 6.5*10**(-6), 0])
        #E.append([7500.0, 0.45, B_F[0], -0.0856, 6.5*10**(-6), 0])
        # ========================================================================================#

        # Apply extra information about anisotropic, non-linear, no-tension analysis
        analysis = []
        # ========================================================================================#
        '''
            Required Inputs for analysis requirement:
            Args:
                @ analysis(list): A list of all the analysis requirement of dimension (num_layer)
        '''
        analysis.append([0, 0, 0])# analysis[0]->anisotropic; analysis[1]->nonlinear; analysis[2]->no-tension; analysis[3]->viscoelastic or not
        #analysis.append([0, 0, 0])# 0->Needed; 1->Do not need
        #analysis.append([0, 0, 0])
        # analysis.append([0, 0, 0])
        # analysis.append([0, 0, 0])
        # ========================================================================================#

        # Apply load vector and the specimen dimension (Test or Pavement)
        # ========================================================================================#
        '''
            Required Inputs for Test method selection:
            Args:
                @ Test_method(str): A string with option 'Test' or 'Pavement'
        '''
        Test_method = 'Pavement'
        # ========================================================================================#

        '''
            Required Inputs for loading information when choosing 'Test':
            Args:
                @ x_len(float64): The length of the surface as well as the load (the same length)
                @ F(float64): The load that applied at the surface, negative: up and positive: down
                @ confined(0 or 1): Confining condition for the specimen,
                0->no fully confining; 1->fully confining.
                @ partial_confined: Confining condition for the specimen,
                0->no partial confining; 1->partial confining.
                @ F_confined(float64): The applied fully confining pressure
                @ F_confined_coor(list): The range of partial confining pressure with dimension of 2
            All the rest values should not be modified!
        '''

        # name = str(x_coor) + '_' + str(y_coor) + "_input.txt"
        name = "results/input_" + str(num) + ".txt"
        if (Test_method == 'Test'):
            full_surface_load = 1
        # ========================================================================================#
            x_len = 75 * 1 / 25.4
            F = - sigma1[file] # negative: down and positive: up
            confined = 1
            if (confined):
                F_confined = - sigma3[file] # negative: left and positive: right
            partial_confined = 0
            if (partial_confined):
                F_confined = -20.8*0.145038
                F_confined_coor = [0, 0]
        # ========================================================================================#
        '''
            Required Inputs for loading information when choosing 'Pavement':
            Assume the surface length is 10 times of the load range.
            Args:
                @ F_x(float64): The length of the load.
                @ F(float64): The load that applied at the surface, negative: up and positive: down
            All the rest values should not be modified!
        '''
        if (Test_method == 'Pavement'):
            full_surface_load = 0
        # ========================================================================================#
            F_x = radius
            F = -force # negative: down and positive: up
        # ========================================================================================#
            if x_len < 0:
                x_len = 20*F_x

        # Apply prescribed displacement
        '''
            Required Inputs for prescirbed displacements:
            Args:
                @ left_boundary(int): 0: no fix, 1: all fix, 2: x fix, 3: y fix
                @ right_boundary(int): 0: no fix, 1: all fix, 2: x fix, 3: y fix
                @ bottom_boundary(int): 0: no fix, 1: all fix, 2: x fix, 3: y fix
        '''
        # ========================================================================================#
        left_boundary = 2
        right_boundary = 2
        bottom_boundary = 1
        # ========================================================================================#

        # Apply the density of both x and y direction
        '''
            Required Inputs for mesh density:
            Args:
                @ x_num(int): x-direction density
                @ y_num(int): y-direction density
        '''
        # ========================================================================================#
        x_num = 15 #15
        y_num = 30 #9
        # ========================================================================================#

        # Set up the name for the output file
        '''
            Args:
                @ name(str): Name for the output file. Need to have the suffix. Ex: "input.txt".
        '''
        # ========================================================================================#
        #name = "input_2.txt"
        # ========================================================================================#


        '''
        The following part should not be modified!!
        '''
        # ==================================Do Not Modified This Part !===========================#
        # X-direction mesh
        if full_surface_load:
            X = [0, x_len]
            surface_mesh = np.linspace(X[0], X[1], x_num, endpoint=True)
        else:
            X = [0, F_x, x_len]
            surface_mesh1 = np.linspace(X[0], X[1], int(x_num/3), endpoint=False)
            #surface_mesh2 = np.logspace(np.log10(X[1]), np.log10(X[2]), int(2*x_num/3))
            surface_mesh2 = np.linspace(X[1], X[2], 30, endpoint=True)
            surface_mesh = np.append(surface_mesh1, surface_mesh2)

        # Y-direction mesh
        Y = [0]
        for i in range(num_layer):
            Y.append(-np.sum(layer_thickness[0:i+1]))
        layer_mesh = np.zeros(0)
        for i in range(num_layer):
            layer_mesh = np.append(layer_mesh,np.linspace(Y[i],Y[i+1],50,endpoint=False))
#            layer_mesh = np.append(layer_mesh,np.linspace(Y[i],Y[i+1],int((num_layer-i)**0.2*y_num/2),endpoint=False))
#            temp = np.logspace(np.log(Y[i]+1)/np.log(1.5),np.log(-Y[i+1]+1)/np.log(1.5),20,endpoint=False,base=1.5)
#            layer_mesh = np.append(layer_mesh,-temp + 1)
            layer_mesh = np.append(layer_mesh,np.ones(1)*Y[-1])
        #print("surface mesh: ", surface_mesh)
        #print("layer mesh: ", layer_mesh)

        # Cumulatively count the elements number
        temp = []
        elem_num = [0]
        for i in range(num_layer):
            temp.append((surface_mesh.shape[0]-1)*int((num_layer-i)**0.2*y_num/2))
        for i in range(num_layer):
            element_count = 0
            for j in range(i+1):
                element_count += temp[j]
            elem_num.append(element_count)

        # Generate Mesh
        x = surface_mesh
        y = layer_mesh
        num_elem = (x.shape[0]-1)*(y.shape[0]-1)
        num_node = (x.shape[0]*2-1)*(y.shape[0]*2-1) - num_elem
        # Another equivalent way: num_node = (x.shape[0]*2-1)*y.shape[0] + x.shape[0]*(y.shape[0] - 1)

        # Node coordinates
        node = {} # A dictionary (node index: node coordinates) Ex:{0: (0,0)}
        node_re = {} # A dictionary (node coordinates: node index) Ex:{(0,0): 0}
        elem_cen_re ={} # A dictionary (element center: element index) Ex:{(0.5,0.5): 1}
        idx = 0 # node index
        idx0 = 0 # element index
        for j in range(2*y.shape[0]-1):
            for i in range(2*x.shape[0]-1):
                if (i%2 == 1 and j%2 != 1):
                    x_temp = (x[int((i+1)/2)]+x[int((i-1)/2)])/2
                    y_temp = y[int(j/2)]
                    node[idx] = (x_temp, y_temp)
                    node_re[x_temp, y_temp] = idx
                    idx += 1
                elif (i%2 != 1 and j%2 == 1):
                    y_temp = (y[int((j+1)/2)]+y[int((j-1)/2)])/2
                    x_temp = x[int(i/2)]
                    node[idx] = (x_temp, y_temp)
                    node_re[x_temp, y_temp] = idx
                    idx += 1
                elif (i%2 != 1 and j%2 != 1):
                    x_temp = x[int(i/2)]
                    y_temp = y[int(j/2)]
                    node[idx] = (x_temp, y_temp)
                    node_re[x_temp, y_temp] = idx
                    idx += 1
                elif (i%2 == 1 and j%2 == 1):
                    x_temp = (x[int((i+1)/2)]+x[int((i-1)/2)])/2
                    y_temp = (y[int((j+1)/2)]+y[int((j-1)/2)])/2
                    elem_cen_re[x_temp, y_temp] = idx0
                    idx0 += 1
        elem = {} # A dictionary (element index: [element node list]) Ex:(0: [1, 2, 3, 4, 5, 6, 7, 8])
        for j in range(2*y.shape[0]-1):
            for i in range(2*x.shape[0]-1):
                temp = []
                if (i%2 == 1 and j%2 == 1):
                    xtemp = x[int((i-1)/2)]
                    ytemp = y[int((j+1)/2)]
                    temp.append(node_re[xtemp,ytemp])

                    xtemp = x[int((i+1)/2)]
                    ytemp = y[int((j+1)/2)]
                    temp.append(node_re[xtemp,ytemp])

                    xtemp = x[int((i+1)/2)]
                    ytemp = y[int((j-1)/2)]
                    temp.append(node_re[xtemp,ytemp])

                    xtemp = x[int((i-1)/2)]
                    ytemp = y[int((j-1)/2)]
                    temp.append(node_re[xtemp,ytemp])

                    xtemp = (x[int((i+1)/2)]+x[int((i-1)/2)])/2
                    ytemp = y[int((j+1)/2)]
                    temp.append(node_re[xtemp,ytemp])

                    xtemp = x[int((i+1)/2)]
                    ytemp = (y[int((j+1)/2)]+y[int((j-1)/2)])/2
                    temp.append(node_re[xtemp,ytemp])

                    xtemp = (x[int((i+1)/2)]+x[int((i-1)/2)])/2
                    ytemp = y[int((j-1)/2)]
                    temp.append(node_re[xtemp,ytemp])

                    xtemp = x[int((i-1)/2)]
                    ytemp = (y[int((j+1)/2)]+y[int((j-1)/2)])/2
                    temp.append(node_re[xtemp,ytemp])

                    xtemp = (x[int((i+1)/2)]+x[int((i-1)/2)])/2
                    ytemp = (y[int((j+1)/2)]+y[int((j-1)/2)])/2
                    idx = elem_cen_re[xtemp, ytemp]
                    elem[idx] = temp

        # Fill in the node indices (there is one node at the middle of on edge of a quadrilateral)
        x_full = []
        y_full = []

        # Fill x-direction node indices
        for i in range(x.shape[0]-1):
            x_full.append(x[i])
            x_full.append((x[i]+x[i+1])/2)
        x_full.append(x[-1])
        x_full = np.array(x_full)

        # Fill y-direction node indices
        for i in range(y.shape[0]-1):
            y_full.append(y[i])
            y_full.append((y[i]+y[i+1])/2)
        y_full.append(y[-1])
        y_full = np.array(y_full)

        # Apply the force vector
        y_force_edge = {}
        x_force_edge = {}
        if (Test_method == 'Test'):
            # Apply surface load
            for i in range(len(x) - 1):
                temp = elem_cen_re[x[i]/2+x[i+1]/2, y[0]/2+y[1]/2]
                y_force_edge[temp] = F
            # Apply confining pressure if needed
            if (confined == 1):
                for j in range(len(y) - 1):
                    temp = elem_cen_re[x[-1]/2+x[-2]/2, y[j]/2+y[j+1]/2]
                    x_force_edge[temp] = F_confined
            if (partial_confined == 1): # Be careful of the sign, all the y coordinates start from 0 to negative
                start = F_confined_coor[0]
                end = F_confined_coor[1]
                for j in range(len(y) - 1):
                    if (y[j] >= start and y[j+1] < start):
                        y_start = j
                    if (y[j] >= end and y[j+1] < end):
                        y_end = j
                for j in range(y_end - y_start + 1):
                    m = j + y_start
                    temp = elem_cen_re[x[-1]/2+x[-2]/2, y[m]/2+y[m+1]/2]
                    x_force_edge[temp] = F_confined
        if (Test_method == 'Pavement'):
            # Apply surface load
            for i in range(int(x_num/3)):
                temp = elem_cen_re[x[i]/2+x[i+1]/2, y[0]/2+y[1]/2]
                y_force_edge[temp] = F

        # Apply the x-direction prescribed displacement
        pre_xdisp = {}
        if (left_boundary == 1 or left_boundary == 2):
            for i in range(len(y_full)):
                pre_xdisp[node_re[x_full[0], y_full[i]]] = 0
        if (right_boundary == 1 or right_boundary == 2):
            for i in range(len(y_full)):
                pre_xdisp[node_re[x_full[-1], y_full[i]]] = 0
        if (bottom_boundary == 1 or bottom_boundary == 2):
            for i in range(len(x_full)):
                pre_xdisp[node_re[x_full[i], y_full[-1]]] = 0

        # Apply the y-direction prescribed displacement
        pre_ydisp = {}
        if (left_boundary == 1 or left_boundary == 3):
            for i in range(len(y_full)):
                pre_ydisp[node_re[x_full[0], y_full[i]]] = 0
        if (right_boundary == 1 or right_boundary == 3):
            for i in range(len(y_full)):
                pre_ydisp[node_re[x_full[-1], y_full[i]]] = 0
        if (bottom_boundary == 1 or bottom_boundary == 3):
            for i in range(len(x_full)):
                pre_ydisp[node_re[x_full[i], y_full[-1]]] = 0

        # Output as an input.txt file
        # Output file name
        f = open(name,"w")

        # Output essential keys
        # (number of nodes, number of layers, number of x-direction point load, y-direction point load...
        #  edge load, x_prescribed disp., y_prescribed disp.)
        f.write("%d %d %d %d %d %d %d %d\r\n" %\
                (num_node, num_elem, num_layer, 0, 0, len(x_force_edge)+len(y_force_edge), len(pre_xdisp), len(pre_ydisp)))

        # Output layer information
        for i in range(num_layer):
            f.write("%d %d %d %d %d"%(elem_num[i], elem_num[i+1]-1, analysis[i][0], analysis[i][1], analysis[i][2]))
            f.write("\r\n")
            for j in range(len(E[i])):
                f.write("%f"%(E[i][j]))
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
        '''
        # Output y-direction edge load
        for i in y_force_edge:
            f.write("%d %d\r\n"%(i,2))
            f.write("%.4f %.4f\r\n"%(0,y_force_edge[i]))

        # Output x-direction edge load
        for i in x_force_edge:
            f.write("%d %d\r\n"%(i,1))
            f.write("%.4f %.4f\r\n"%(x_force_edge[i],0))


        # Output node information
        for i in range(num_node):
            f.write("%.4f %.4f\r\n" %(node[i][0], node[i][1]))

        # # Output layer information
        # for i in range(num_layer):
        #     f.write("%d %d %d %d %d"%(elem_num[i], elem_num[i+1]-1, analysis[i][0], analysis[i][1], analysis[i][2]))
        #     f.write("\r\n")
        #     for j in range(len(E[i])):
        #         f.write("%f"%(E[i][j]))
        #         f.write("%s"%" ")
        #     f.write("\r\n")

        # # Output model information
        # f.write("%d\r\n"%(model_num))
        # for i in range(len(model_para)):
        #     f.write("%.4f "%(model_para[i]))
        # f.write("\r\n")
        # for i in range(len(iter_para)):
        #     f.write("%.4f "%(iter_para[i]))
        # f.write("\r\n")

        # Output element information
        for i in range(num_elem):
            f.write("%d %d %d %d %d %d %d %d %d\r\n" %\
                (8,elem[i][0],elem[i][1],elem[i][2],elem[i][3],elem[i][4],elem[i][5],elem[i][6],elem[i][7]))

        # # Output y-direction edge load
        # for i in y_force_edge:
        #     f.write("%d %d\r\n"%(i,2))
        #     f.write("%.4f %.4f\r\n"%(0,y_force_edge[i]))

        # # Output x-direction edge load
        # for i in x_force_edge:
        #     f.write("%d %d\r\n"%(i,1))
        #     f.write("%.4f %.4f\r\n"%(x_force_edge[i],0))

        # Output x-direction prescribed disp.
        for i in pre_xdisp:
            f.write("%d "%i)
        if (len(pre_xdisp)!=0):
            f.write("\r\n")
        for i in pre_xdisp:
            f.write("%.5f "%pre_xdisp[i])
        if (len(pre_xdisp)!=0):
            f.write("\r\n")

        # Output y-direction prescribed disp.
        for i in pre_ydisp:
            f.write("%d "%i)
        if (len(pre_ydisp)!=0):
            f.write("\r\n")
        for i in pre_ydisp:
            f.write("%.5f "%pre_ydisp[i])
        f.close()
        # ==================================Do Not Modified This Part !===========================#
        return name

# meshgenerator = Mesh_generator()
# database = Database(vehicle_type = 'EC-135A')
# dic_vehicle = database.query()
# output_field = database.output_field
#
# num_input = len(dic_vehicle[output_field[0]])
# print("Total number of input files:", num_input)
#
# for i in range(num_input):
#     force = dic_vehicle['pressure'][i]
#     area = dic_vehicle['contactarea'][i]
#     meshgenerator.generate_mesh(force, area, i)
