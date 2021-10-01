from enum import Enum


class UnaryStatesEnum(Enum):
    Burnt = 0
    CleaningTool = 1
    HeatSourceOrSink = 2
    Cooked = 3
    Dusty = 4
    Frozen = 5
    Open = 6
    Sliced = 7
    Slicer = 8
    Soaked = 9
    Stained = 10
    ToggledOn = 11
    WaterSource = 12
    InFOVOfRobot = 13
    InHandOfRobot = 14
    InReachOfRobot = 15
    InSameRoomAsRobot = 16


# Temperture and max temperature are not included


class BinaryStatesEnum(Enum):
    InRoom = 0
    Inside = 1
    NextTo = 2
    OnFloor = 3
    OnTop = 4
    Touching = 5
    Under = 6


# Not in use, we use binary InRoom for these
class RoomStatesEnum(Enum):
    IsInBathroom = 0
    IsInBedroom = 1
    IsInChildsRoom = 2
    IsInCloset = 3
    IsInCorridor = 4
    IsInDiningRoom = 5
    IsInEmptyRoom = 6
    IsInExerciseRoom = 7
    IsInGarage = 8
    IsInHomeOffice = 9
    IsInKitchen = 10
    IsInLivingRoom = 11
    IsInLobby = 12
    IsInPantryRoom = 13
    IsInPlayroom = 14
    IsInStaircase = 15
    IsInStorageRoom = 16
    IsInTelevisionRoom = 17
    IsInUtilityRoom = 18
    IsInBalcony = 19
    IsInLibrary = 20
    IsInAuditorium = 21
    IsInUndefined = 22
