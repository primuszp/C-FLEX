"""
Class Database for MySQL querying U.S. Army ERDC's vehicle database.

Copyright (c) 2019 Haohang Huang
Licensed under the GPL License (see LICENSE for details)
Written by Haohang Huang, November 2019.

Usage: python database.py
"""

"""
Description of ARMY-ERDC's PSEVEN database (See schema.sql for details):
CREATE TABLE `Vehicles`
 (
	`VehicleId`			varchar (40),
	`Name`			varchar (240),
	...
);
CREATE TABLE `GearTires`
 (
	`VehicleId`			varchar (40),
	`PercentOfLoad`			float NOT NULL,
	`xCoordinate`			float,
	`yCoordinate`			float,
	`Pressure`			float,
	`ContactArea`			float,
	`Shape`			float,
	`UseForStress`			char NOT NULL,
	`UseForEswl`			char NOT NULL,
	`UseForPcr`			char NOT NULL,
	`UseForAcn`			char NOT NULL,
	`UseForLed`			char NOT NULL,
	`NoseGearTire`			char NOT NULL,
	`FixPressure`			char NOT NULL,
	`ShapeType`			int
);
CREATE TABLE `EvaluationPoints`
 (
	`VehicleId`			varchar (40),
	`xCoordinate`			float,
	`yCoordinate`			float,
	`zCoordinate`			float
);
"""

import pymysql # pip install pymysql

class Database():

    def __init__(self, name, hostname = 'localhost', username = 'root', password = ''):
        """Connect to a MySQL database, by default user = root, password = ''. Change the values according to the MySQL settings on your machine.
        """
        self.name = name
        self.connection = pymysql.connect(
            db=name,
            host=hostname,
            user=username,
            passwd=password,
            cursorclass=pymysql.cursors.DictCursor
        )
        print("> Database '{}' successfully linked".format(name))

        self.vehicleList = None

    def vehicle_all(self):
        """Get a complete list of all vehicle names (sorted in alphabetical order b.c. PSEVEN software does this).
        """
        vehicle_names = []
        with self.connection.cursor() as cursor:
            cursor.execute("select name from vehicles;")
            for row in cursor.fetchall():
                vehicle_names.append(row['name'])
        self.vehicleList = sorted(vehicle_names)
        return self.vehicleList

    def query(self, fields, vehicle=None):
        """Query vehicle info from the database.
        Args:
            fields [list<str>]: field names you want to query for the vehicle
            vehicle [str/int]: name or 1-based index of the query vehicle.
        Returns:
            [list<dict>]: vehicle info [ {Tire1}, {Tire2}, {Tire3}, ...]
        """

        # get vehicle name if an index is given
        if isinstance(vehicle, int): # if 1-based index
            if self.vehicleList is None: # ensure we have the full list
                self.vehicle_all()
            vehicle = self.vehicleList[vehicle - 1] # 0-based index

        with self.connection.cursor() as cursor:
            q = "select * from geartires where vehicleid = (select vehicleid from vehicles where name = '{}');"
            cursor.execute(q.format(vehicle))
            result = cursor.fetchall() # a list of dictionary (each element is a row)
            if (len(result) == 0):
                print("Vehicle {} not found!".format(vehicle))
            for i, row in enumerate(result):
                info = { f : row[f] for f in fields } # extract fields
                result[i] = info # overwrite
        return result

    def query_evaluation(self, vehicle):
        """Query evaluation points of the vehicle.
        Args:
            vehicle [str]: name of the query vehicle.
        Returns:
            [list<dict>]: vehicle info [ {Tire1}, {Tire2}, {Tire3}, ...]
        """
        with self.connection.cursor() as cursor:
            q = "select * from evaluationpoints where vehicleid = (select vehicleid from vehicles where name = '{}');"
            cursor.execute(q.format(vehicle))
            result = cursor.fetchall() # a list of dictionary (each element is a row)
            if (len(result) == 0):
                print("Vehicle {} not found!".format(vehicle))
        return result

    def close(self):
        self.connection.close()
        print("> Database '{}' gracefully closed".format(self.name))

if __name__ == "__main__":
    import numpy as np
    from plot import plot_tire_eval

    # === Connect === #
    db = Database(name='erdc')

    # === Get all vehicles in alphabetical order (see PSEVEN) === #
    vehicle_list = db.vehicle_all()
    print("> Complete list of vehicles:")
    for i, v in enumerate(vehicle_list):
        print("[{:03}]: {}".format(i+1, v)) # note the 1-based index
    print("")

    # === Query one vehicle (by name or by ID) === #
    vehicle = 'Boeing 777-300'
    # vehicle = 124
    num_tires = 6 # see config.py

    # ID to name
    if isinstance(vehicle, int):
        vehicle = vehicle_list[vehicle-1]

    # Tire configuration
    print("> Gear configuration of vehicle {}:".format(vehicle))
    fields = ['PercentOfLoad','xCoordinate','yCoordinate','Pressure','ContactArea']
    tires = db.query(fields=fields, vehicle=vehicle)

    print("{:10}".format("Tire"), *["{:15}".format(f) for f in fields])
    for i, r in enumerate(tires):
        print("{:10}".format(str(i)), *["{:15}".format(str(r[f])) for f in fields])
    print("")

    # Evaluation points
    print("> Evaluation points of vehicle {}:".format(vehicle))
    evals = db.query_evaluation(vehicle=vehicle)
    pts = np.zeros((len(evals), 2))
    print("{:8} {:8} {:8} {:8}".format("Point", "X", "Y", "Z"))
    for i, r in enumerate(evals):
        pts[i] = [r['xCoordinate'], r['yCoordinate']]
        print("{:<8} {:<8.1f} {:<8.1f} {:<8.1f}".format(i, r['xCoordinate'], r['yCoordinate'], r['zCoordinate']))

    # Truncate
    xy = np.zeros((len(tires), 2))
    force, area = np.zeros(len(tires)), np.zeros(len(tires))
    for i, tire in enumerate(tires):
        _, X, Y, F, A = [tire[f] for f in fields]
        xy[i] = [X, Y]
        force[i], area[i] = F, A
    radius = np.sqrt(area / np.pi)

    if num_tires != -1:
        xy, force, area, radius = xy[:num_tires,:], force[:num_tires], area[:num_tires], radius[:num_tires]
    else:
        num_tires = len(tires) # update num_tires

    # Plot
    plot_tire_eval(xy, radius, pts, vehicle=vehicle) # just plot
    # plot_tire_eval(xy, radius, pts, vehicle=vehicle, path='./test.png') # save figure

    # === Close === #
    db.close()
