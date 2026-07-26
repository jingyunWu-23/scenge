# Description: The ego vehicle approaches a slow leading car and attempts to pass using an adjacent lane while an adversarial car in the passing lane constrains the available gap.
# AdvType: Car
# AdvPos: In the adjacent passing lane
# AdvBehavior: Constrains the available passing gap
Town = globalParameters.town
param map = localPath(f'../maps/{Town}.xodr')
param carla_map = Town

model scenic.simulators.carla.model

EgoSpawnPt = OrientedPoint at globalParameters.spawnPt,
    with heading globalParameters.yaw deg

Waypoints = globalParameters.waypoints
LanePts = globalParameters.lanePts

EGO_MODEL = 'vehicle.lincoln.mkz_2017'

param OPT_LONG_DIST = Range(18, 28)
param OPT_ADV_LONG_DIST = Range(30, 42)
param OPT_LAT_DIST = Range(3.2, 3.8)
param OPT_TRIGGER_DIST = Range(8, 18)

ego = Car at EgoSpawnPt,
    with regionContainedIn None,
    with blueprint EGO_MODEL

LeadingPt = OrientedPoint following roadDirection from EgoSpawnPt for globalParameters.OPT_LONG_DIST

behavior SlowForward():
    take SetThrottleAction(0.3), SetSteerAction(0)

leadingCar = Car at LeadingPt,
    with heading LeadingPt.heading,
    with regionContainedIn None,
    with behavior SlowForward()

IntSpawnPt = OrientedPoint following roadDirection from EgoSpawnPt for globalParameters.OPT_ADV_LONG_DIST

behavior AdvBehavior():
    while ego.distanceTo(self) > globalParameters.OPT_TRIGGER_DIST:
        take SetThrottleAction(0.5), SetSteerAction(0)
    take SetBrakeAction(0.8), SetThrottleAction(0)

AdvAgent = Car left of IntSpawnPt by globalParameters.OPT_LAT_DIST,
    with heading IntSpawnPt.heading,
    with regionContainedIn None,
    with behavior AdvBehavior()
