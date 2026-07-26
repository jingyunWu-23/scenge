# Description: The ego vehicle drives on a straight lane behind an adversarial leading car that brakes hard when the ego closes the gap.
# AdvType: Car
# AdvPos: Initially ahead in the same lane
# AdvBehavior: Hard brake
Town = globalParameters.town
param map = localPath(f'../maps/{Town}.xodr')
param carla_map = Town

model scenic.simulators.carla.model

EgoSpawnPt = OrientedPoint at globalParameters.spawnPt,
    with heading globalParameters.yaw deg

waypoints = globalParameters.waypoints
lanePts = globalParameters.lanePts

EGO_MODEL = 'vehicle.lincoln.mkz_2017'

param OPT_GEO_X_DISTANCE = Range(-5.0, 5.0)
param OPT_GEO_Y_DISTANCE = Range(-2.0, 2.0)
param OPT_LONGITUDINAL_DISTANCE = Range(15.0, 30.0)
param OPT_EGO_SPEED = Range(10.0, 20.0)
param OPT_ADV_SPEED = Range(5.0, 15.0)
param OPT_TRIGGER_DISTANCE = Range(5.0, 15.0)
param OPT_BRAKE_DECEL = Range(5.0, 10.0)
param OPT_TIMING_DELAY = Range(0.1, 1.0)

ego = Car at EgoSpawnPt,
    with regionContainedIn None,
    with blueprint EGO_MODEL

IntSpawnPt = OrientedPoint following roadDirection from EgoSpawnPt for globalParameters.OPT_GEO_Y_DISTANCE

behavior WaitForTrigger():
    while distance(self, ego) > globalParameters.OPT_TRIGGER_DISTANCE:
        wait

behavior AdvBehavior():
    while True:
        take SetSpeedAction(globalParameters.OPT_ADV_SPEED)
        do WaitForTrigger()
        wait
        take SetDecelerationAction(globalParameters.OPT_BRAKE_DECEL)

AdvAgent = Car left of IntSpawnPt by globalParameters.OPT_GEO_X_DISTANCE,
    with heading IntSpawnPt.heading,
    with regionContainedIn None,
    with behavior AdvBehavior()
