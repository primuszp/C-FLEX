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

    def close(self):
        self.connection.close()
        print("> Database '{}' gracefully closed".format(self.name))

if __name__ == "__main__":
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

    fields = ['PercentOfLoad','xCoordinate','yCoordinate','Pressure','ContactArea']
    results = db.query(fields=fields, vehicle=vehicle)

    print("> Query results for vehicle {}:".format(vehicle if not isinstance(vehicle,int) else vehicle_list[vehicle-1]))
    print("{:10}".format("Tire"), *["{:15}".format(f) for f in fields])
    for i, r in enumerate(results):
        print("{:10}".format(str(i)), *["{:15}".format(str(r[f])) for f in fields])

    # === Close === #
    db.close()
