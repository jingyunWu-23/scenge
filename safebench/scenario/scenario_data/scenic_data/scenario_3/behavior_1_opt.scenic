# Description: The ego vehicle is blocked by a slow leading car in its current lane and faces risky lane-change pressure from a front vehicle and a faster rear vehicle in the target lane.
# AdvType: Car
# AdvPos: A slow leading car in the ego's current lane, and a front vehicle and faster rear vehicle in the adjacent target lane.
# AdvBehavior: Slow leading vehicle and target-lane front and rear vehicles blocking the lane change.
Town = globalParameters.town
param map = localPath(f'../maps/{Town}.xodr')
param carla_map = Town

model scenic.simulators.carla.model

EgoSpawnPt = OrientedPoint at globalParameters.spawnPt,
    with heading globalParameters.yaw deg

waypoints = globalParameters.waypoints
lanePts = globalParameters.lanePts

EGO_MODEL = 'vehicle.lincoln.mkz_2017'

param OPT_LONGITUDINAL_DISTANCE = Range(10, 30)
param OPT_LATERAL_DISTANCE = Range(3, 5)
param OPT_SPEED = Range(5, 15)
param OPT_TRIGGER_DISTANCE = Range(10, 20)
param OPT_TIMING = Range(1, 5)
param OPT_FRONT_DISTANCE = Range(15, 25)
param OPT_REAR_DISTANCE = Range(10, 20)
param OPT_FRONT_SPEED = Range(15, 25)
param OPT_REAR_SPEED = Range(20, 30)

behavior AdvBehaviorLead():
    take SetSpeedAction(globalParameters.OPT_SPEED)
    while distance(self, ego) > globalParameters.OPT_TRIGGER_DISTANCE:
        wait
    wait
    take SetSpeedAction(0)

behavior AdvBehaviorFront():
    take SetSpeedAction(globalParameters.OPT_FRONT_SPEED)

behavior AdvBehaviorRear():
    take SetSpeedAction(globalParameters.OPT_REAR_SPEED)

ego = Car at EgoSpawnPt,
    with regionContainedIn None,
    with blueprint EGO_MODEL

LeadPt = OrientedPoint following roadDirection from EgoSpawnPt for globalParameters.OPT_LONGITUDINAL_DISTANCE
AdvAgentLead = Car at LeadPt,
    with heading LeadPt.heading,
    with regionContainedIn None,
    with behavior AdvBehaviorLead()

TargetFrontPt = OrientedPoint following roadDirection from EgoSpawnPt for globalParameters.OPT_FRONT_DISTANCE
AdvAgentFront = Car left of TargetFrontPt by globalParameters.OPT_LATERAL_DISTANCE,
    with heading TargetFrontPt.heading,
    with regionContainedIn None,
    with behavior AdvBehaviorFront()

TargetRearPt = OrientedPoint following roadDirection from EgoSpawnPt for -globalParameters.OPT_REAR_DISTANCE
AdvAgentRear = Car left of TargetRearPt by globalParameters.OPT_LATERAL_DISTANCE,
    with heading TargetRearPt.heading,
    with regionContainedIn None,
    with behavior AdvBehaviorRear()
