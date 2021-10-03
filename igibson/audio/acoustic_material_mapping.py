from igibson.audio.audio_system import AcousticMesh
import numpy as np
import csv
from igibson.simulator import Simulator
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.objects import cube

#This is an inelegant (and potentially unsustainable) way to map materials, and should be directly unified
#with the C enums in resonance audio 

ResonanceMaterialToId = {
  'Transparent' : 0,
  'AcousticCeilingTiles' : 1,
  'BrickBare' : 2,
  'BrickPainted' : 3,
  'ConcreteBlockCoarse' : 4,
  'ConcreteBlockPainted' : 5,
  'CurtainHeavy' : 6,
  'FiberGlassInsulation' : 7,
  'GlassThin' : 8,
  'GlassThick' : 9,
  'Grass' : 10,
  'LinoleumOnConcrete' : 11,
  'Marble' : 12,
  'Metal': 13,
  'ParquetOnConcrete' : 14,
  'PlasterRough' : 15,
  'PlasterSmooth' : 16,
  'PlywoodPanel' : 17,
  'PolishedConcreteOrTile': 18,
  'Sheetrock' : 19,
  'WaterOrIceSurface' : 20,
  'WoodCeiling' : 21,
  'WoodPanel' : 22,
  'Uniform' : 23
}

IdToResonanceMaterial = {v : k for k, v in ResonanceMaterialToId.items() }
