-- ----------------------------------------------------------
-- MDB Tools - A library for reading MS Access database files
-- Copyright (C) 2000-2011 Brian Bruns and others.
-- Files in libmdb are licensed under LGPL and the utilities under
-- the GPL, see COPYING.LIB and COPYING files respectively.
-- Check out http://mdbtools.sourceforge.net
-- ----------------------------------------------------------

-- That file uses encoding UTF-8

CREATE TABLE `Characteristics`
 (
	`VehicleId`			varchar (100) NOT NULL, 
	`Manufacturer`			varchar (100), 
	`EngineMan`			varchar (100), 
	`EngineNum`			int, 
	`EngineModel`			varchar (100), 
	`LandGearType`			varchar (100), 
	`HoverCeilingIn`			float, 
	`HoverCeilingOut`			float, 
	`TurningRadius`			float, 
	`PlumeLength`			float, 
	`WingToWingPark`			float, 
	`WingWalkerC`			float, 
	`MaxMunition`			float, 
	`PaveWidth180`			float, 
	`WingtipMunition`			char NOT NULL, 
	`ReverseThrust`			char NOT NULL, 
	`WingSpan`			float, 
	`OverallLength`			float, 
	`OverallHeight`			float, 
	`VerticalWingtipC`			float, 
	`VerticalTailC`			float, 
	`SkidLength`			float, 
	`RearToTail`			float, 
	`FuselageToTail`			float, 
	`TailRotorDiameter`			float, 
	`TopRotorHeadHeight`			float, 
	`MinWingC`			float, 
	`CenterToOutboard`			float, 
	`NoseToRearWheel`			float, 
	`FootprintSide45`			float, 
	`MinTurningRadius`			float, 
	`EngineDangerIdle`			varchar (100), 
	`GroundC`			float, 
	`MaxKVA`			float, 
	`AirStarting`			varchar (100), 
	`PreconAir`			varchar (100), 
	`EnvironAir`			varchar (100), 
	`MinTakeWeight`			float, 
	`MinTakeDistMin`			float, 
	`MinTakeDistMinAbort`			float, 
	`MaxTakeWeightPeace`			float, 
	`MaxTakeWeightWar`			float, 
	`MinTakeDistMax`			float, 
	`MinRunwayWidth`			float, 
	`MinTaxiwayWidth`			float, 
	`MinLandingWeight`			float, 
	`MaxLandingWeight`			float, 
	`MinLandDistMin`			float, 
	`MinLandDistMax`			float, 
	`PeaceTakeWeight`			float, 
	`PeaceLandingWeight`			float, 
	`WingToWingTaxiway`			float, 
	`WingToNoseTaxiway`			float, 
	`WingToTailTaxiway`			float
);

CREATE TABLE `EvaluationPoints`
 (
	`VehicleId`			varchar (40), 
	`xCoordinate`			float, 
	`yCoordinate`			float, 
	`zCoordinate`			float
);

CREATE TABLE `Pictures`
 (
	`VehicleId`			varchar (40), 
	`Picture`			text (255)
);

CREATE TABLE `Setup`
 (
	`ShowVehicles`			int, 
	`LastDate`			datetime
);

CREATE TABLE `SlopeIntercepts`
 (
	`VehicleId`			varchar (40), 
	`SlopeA`			float NOT NULL, 
	`SlopeB`			float, 
	`SlopeC`			float, 
	`SlopeD`			float, 
	`InterceptA`			float, 
	`InterceptB`			float, 
	`InterceptC`			float NOT NULL, 
	`InterceptD`			float NOT NULL, 
	`Flexible`			char NOT NULL
);

CREATE TABLE `Vehicles`
 (
	`VehicleId`			varchar (40), 
	`Name`			varchar (240), 
	`SurfaceThicknessGroupNumber`			int, 
	`BaseThicknessGroupNumber`			int, 
	`StandardLoad`			int, 
	`MinimumLoad`			float, 
	`MaximumLoad`			float, 
	`Comments`			text (255), 
	`Custom`			char NOT NULL, 
	`Aircraft`			char NOT NULL, 
	`TypeOfGroundVehicle`			int, 
	`NumberOfAxles`			int
);

CREATE TABLE `VersionChanges`
 (
	`VersionId`			varchar (40), 
	`ChangeId`			varchar (40), 
	`ObjectType`			int, 
	`ObjectName`			varchar (200), 
	`ElementType`			int, 
	`ElementName`			varchar (200), 
	`ActionType`			int
);

CREATE TABLE `Versions`
 (
	`VersionId`			varchar (40), 
	`VersionDateTime`			datetime
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


