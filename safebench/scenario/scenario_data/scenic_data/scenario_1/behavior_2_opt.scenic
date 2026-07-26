# Description: The ego vehicle drives on a straight lane behind an adversarial leading car that suddenly brakes hard.
# AdvType: Car
# AdvPos: Ahead in the same lane
# AdvBehavior: Hard brake
Town = globalParameters.town
EgoSpawnPt = OrientedPoint at globalParameters.spawnPt,
    with heading globalParameters.yaw deg
Waypoints = globalParameters.waypoints
LanePts = globalParameters.lanePts

EGO_MODEL = 'vehicle.lincoln.mkz_2017'

param map = localPath(f'../maps/{Town}.xodr')
param carla_map = Town

model scenic.simulators.carla.model

param OPT_GEO_X_DISTANCE = Range(15, 30)
param OPT_GEO_Y_DISTANCE = Range(0, 0)
param OPT_EGO_SPEED = Range(10, 15)
param OPT_ADV_SPEED = Range(10, 15)
param OPT_TRIGGER_DISTANCE = Range(5, 15)
param OPT_BRAKE_TIME = Range(2, 5)

behavior EgoBehavior():
    while True:
        self.throttle = 1.0
        wait

behavior AdvBehavior():
    while distance(self, ego) > globalParameters.OPT_TRIGGER_DISTANCE:
        self.throttle = 1.0
        wait
    self.brake = 1.0
    self.throttle = 0.0
    wait

ego = Car at EgoSpawnPt,
    with regionContainedIn None,
    with blueprint EGO_MODEL,
    with behavior EgoBehavior()

IntSpawnPt = OrientedPoint following roadDirection from EgoSpawnPt for globalParameters.OPT_GEO_Y_DISTANCE

AdvAgent = Car left of IntSpawnPt by globalParameters.OPT_GEO_X_DISTANCE,
    with heading IntSpawnPt.heading,
    with regionContainedIn None,
    with behavior AdvBehavior()