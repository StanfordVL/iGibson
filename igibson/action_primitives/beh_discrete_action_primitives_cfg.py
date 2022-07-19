from enum import IntEnum

import numpy as np


class BEHActionPrimitiveSet(IntEnum):
    NAVIGATE_TO = 0
    PICK = 1
    PLACE = 2
    TOGGLE = 3
    PULL = 4
    PUSH = 5


ap_object_params = {
    BEHActionPrimitiveSet.NAVIGATE_TO: {  # skill id: move
        "printer.n.03_1": [-0.7, 0, 0, 0],  # dx, dy, dz, target_yaw
        "table.n.02_1": [0, -0.6, 0, 0.5 * np.pi],
        "hamburger.n.01_1": [0, -0.8, 0, 0.5 * np.pi],
        "hamburger.n.01_2": [0, -0.7, 0, 0.5 * np.pi],
        "hamburger.n.01_3": [0, -0.8, 0, 0.5 * np.pi],
        "ashcan.n.01_1": [0, 0.8, 0, -0.5 * np.pi],
        "countertop.n.01_1": [0.0, -0.8, 0, 0.5 * np.pi],  # [0.1, 0.5, 0.8 1.0]
        "pumpkin.n.02_1": [0.45, 0.0, 0.0, 1.0 * np.pi],
        "pumpkin.n.02_2": [0, -0.5, 0, 0.5 * np.pi],
        "cabinet.n.01_1": [0.4, -1.15, 0, 0.5 * np.pi],
    },
    BEHActionPrimitiveSet.PICK: {  # pick
        "printer.n.03_1": [-0.2, 0.0, 0.2],  # dx, dy, dz
        "hamburger.n.01_1": [0.0, 0.0, 0.025],
        "hamburger.n.01_2": [0.0, 0.0, 0.025],
        "hamburger.n.01_3": [0.0, 0.0, 0.025],
        "pumpkin.n.02_1": [0.0, 0.0, 0.0, 1.0],
        "pumpkin.n.02_2": [0.0, 0.0, 0.0, 1.0],
    },
    BEHActionPrimitiveSet.PLACE: {  # place
        "table.n.02_1": [0, 0, 0.5],  # dx, dy, dz
        "ashcan.n.01_1": [0, 0, 0.5],
        "cabinet.n.01_1": [0.3, -0.55, 0.25],
    },
    BEHActionPrimitiveSet.TOGGLE: {  # toggle
        "printer.n.03_1": [-0.3, -0.25, 0.23],  # dx, dy, dz
    },
    BEHActionPrimitiveSet.PULL: {  # pull
        "cabinet.n.01_1": [0.35, -0.3, 0.35, -1, 0, 0, 1],  # pulling_position_offset, pulling_dir (absolute coords)
    },
    BEHActionPrimitiveSet.PUSH: {  # push
        "cabinet.n.01_1": [0.3, -0.6, 0.45, 1, 0, 0],  # dx, dy, dz
    },
}

aps_installing_a_printer = [
    [BEHActionPrimitiveSet.NAVIGATE_TO, "printer.n.03_1"],  # skill id, target_obj
    [BEHActionPrimitiveSet.PICK, "printer.n.03_1"],
    [BEHActionPrimitiveSet.NAVIGATE_TO, "table.n.02_1"],
    [BEHActionPrimitiveSet.PLACE, "table.n.02_1"],
    [BEHActionPrimitiveSet.TOGGLE, "printer.n.03_1"],
]

aps_throwing_away_leftovers = [
    [BEHActionPrimitiveSet.NAVIGATE_TO, "hamburger.n.01_1"],
    [BEHActionPrimitiveSet.PICK, "hamburger.n.01_1"],
    [BEHActionPrimitiveSet.NAVIGATE_TO, "ashcan.n.01_1"],
    [BEHActionPrimitiveSet.PLACE, "ashcan.n.01_1"],  # place
    [BEHActionPrimitiveSet.NAVIGATE_TO, "hamburger.n.01_2"],
    [BEHActionPrimitiveSet.PICK, "hamburger.n.01_2"],
    [BEHActionPrimitiveSet.NAVIGATE_TO, "hamburger.n.01_3"],
    [BEHActionPrimitiveSet.PICK, "hamburger.n.01_3"],
]

aps_putting_leftovers_away = [
    [BEHActionPrimitiveSet.NAVIGATE_TO, "pasta.n.02_1"],
    [BEHActionPrimitiveSet.PICK, "pasta.n.02_1"],
    [BEHActionPrimitiveSet.NAVIGATE_TO, "countertop.n.01_1"],
    [BEHActionPrimitiveSet.PLACE, "countertop.n.01_1"],  # place
    [BEHActionPrimitiveSet.NAVIGATE_TO, "pasta.n.02_2"],
    [BEHActionPrimitiveSet.PICK, "pasta.n.02_2"],
    [BEHActionPrimitiveSet.NAVIGATE_TO, "countertop.n.01_1"],
    [BEHActionPrimitiveSet.PLACE, "countertop.n.01_1"],  # place
    [BEHActionPrimitiveSet.NAVIGATE_TO, "pasta.n.02_2_3"],
    [BEHActionPrimitiveSet.PICK, "pasta.n.02_2_3"],
    [BEHActionPrimitiveSet.NAVIGATE_TO, "countertop.n.01_1"],
    [BEHActionPrimitiveSet.PLACE, "countertop.n.01_1"],  # place
    [BEHActionPrimitiveSet.NAVIGATE_TO, "pasta.n.02_2_4"],
    [BEHActionPrimitiveSet.PICK, "pasta.n.02_2_4"],
    [BEHActionPrimitiveSet.NAVIGATE_TO, "countertop.n.01_1"],
    [BEHActionPrimitiveSet.PLACE, "countertop.n.01_1"],  # place
]

aps_putting_away_Halloween_decorations = [
    [BEHActionPrimitiveSet.NAVIGATE_TO, "cabinet.n.01_1"],  # navigate_to
    [BEHActionPrimitiveSet.PULL, "cabinet.n.01_1"],  # pull_open
    [BEHActionPrimitiveSet.NAVIGATE_TO, "pumpkin.n.02_1"],  # navigate_to
    [BEHActionPrimitiveSet.PICK, "pumpkin.n.02_1"],  # pick
    [BEHActionPrimitiveSet.PLACE, "cabinet.n.01_1"],  # place
    [BEHActionPrimitiveSet.NAVIGATE_TO, "pumpkin.n.02_2"],  # navigate_to
    [BEHActionPrimitiveSet.PICK, "pumpkin.n.02_2"],  # pick
    [BEHActionPrimitiveSet.PUSH, "cabinet.n.01_1"],  # push
]

aps_room_rearrangement = [
    [BEHActionPrimitiveSet.NAVIGATE_TO, "cabinet.n.01_1"],  # move
    [BEHActionPrimitiveSet.PULL, "cabinet.n.01_1"],  # vis pull
    [BEHActionPrimitiveSet.NAVIGATE_TO, "pumpkin.n.02_1"],  # move
    [BEHActionPrimitiveSet.PICK, "pumpkin.n.02_1"],  # vis pick
    [BEHActionPrimitiveSet.PLACE, "cabinet.n.01_1"],  # vis place
    [BEHActionPrimitiveSet.NAVIGATE_TO, "pumpkin.n.02_2"],  # move
    [BEHActionPrimitiveSet.PICK, "pumpkin.n.02_2"],  # vis pick
    [BEHActionPrimitiveSet.PUSH, "cabinet.n.01_1"],  # vis push
]

aps_per_activity = {
    "installing_a_printer": aps_installing_a_printer,
    "throwing_away_leftovers": aps_throwing_away_leftovers,
    "putting_leftovers_away": aps_putting_leftovers_away,
    "putting_away_Halloween_decorations": aps_putting_away_Halloween_decorations,
    "room_rearrangement": aps_room_rearrangement,
}
