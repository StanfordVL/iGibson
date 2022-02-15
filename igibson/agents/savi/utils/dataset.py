# # # for mp3d
# SCENE_SPLITS = {
#     'train': ['sT4fr6TAbpF', 'E9uDoFAP3SH', 'VzqfbhrpDEA', 'kEZ7cmS4wCh', '29hnd4uzFmX', 'ac26ZMwG7aT',
#               'i5noydFURQK', 's8pcmisQ38h', 'rPc6DW4iMge', 'EDJbREhghzL', 'mJXqzFtmKg4', 'B6ByNegPMKs',
#               'JeFG25nYj2p', '82sE5b5pLXE', 'D7N2EKCX4Sj', '7y3sRwLe3Va', 'HxpKQynjfin', '5LpN3gDmAk7',
#               'gTV8FGcVJC9', 'ur6pFq6Qu1A', 'qoiz87JEwZ2', 'PuKPg4mmafe', 'VLzqgDo317F', 'aayBHfsNo7d',
#               'JmbYfDe2QKZ', 'XcA2TqTSSAj', '8WUmhLawc2A', 'sKLMLpTHeUy', 'r47D5H71a5s', 'Uxmj2M2itWa',
#               'Pm6F8kyY3z2', 'p5wJjkQkbXX', '759xd9YjKW5', 'JF19kD82Mey', 'V2XKFyX4ASd', '1LXtFkjw3qL',
#               '17DRP5sb8fy', '5q7pvUzZiYa', 'VVfe2KiqLaN', 'Vvot9Ly1tCj', 'ULsKaCPVFJR', 'D7G3Y4RVNrH',
#               'uNb9QFRL6hY', 'ZMojNkEp431', '2n8kARJN3HM', 'vyrNrziPKCB', 'e9zR4mvMWw7', 'r1Q1Z4BcV1o',
#               'PX4nDJXEHrG', 'YmJkqBEsHnH', 'b8cTxDM8gDG', 'GdvgFV5R1Z5', 'pRbA3pwrgk9', 'jh4fc5c5qoQ',
#               '1pXnuDYAj8r', 'S9hNv5qa7GM', 'VFuaQ6m2Qom', 'cV4RVeZvu5T', 'SN83YJsR3w2'],
#     'val': ['x8F5xyUWy9e', 'QUCTc6BB5sX', 'EU6Fwq7SyZv', '2azQ1b91cZZ', 'Z6MFQCViBuw', 'pLe4wQe7qrG',
#             'oLBMNvg9in8', 'X7HyMhZNoso', 'zsNo4HB9uLZ', 'TbHJrupSAjP', '8194nk5LbLH', 'pa4otMbVnkk', 
#             'yqstnuAEVhm', '5ZKStnWn8Zo', 'Vt2qJdWjCF2', 'wc2JMjhGNzB', 'WYY7iVyf5p8',
#              'fzynW3qQPVF', 'UwV83HsGsw3', 'q9vSo1VnCiC', 'ARNzJeq3xxb', 'rqfALeAoiTq', 'gYvKGZ5eRqb',
#              'YFuZgdQ5vWj', 'jtcxE69GiFV', 'gxdoqLR6rwA'],
# #     'test': ['pa4otMbVnkk', 'yqstnuAEVhm', '5ZKStnWn8Zo', 'Vt2qJdWjCF2', 'wc2JMjhGNzB', 'WYY7iVyf5p8',
# #              'fzynW3qQPVF', 'UwV83HsGsw3', 'q9vSo1VnCiC', 'ARNzJeq3xxb', 'rqfALeAoiTq', 'gYvKGZ5eRqb',
# #              'YFuZgdQ5vWj', 'jtcxE69GiFV', 'gxdoqLR6rwA'],
# }


# for igibson
SCENE_SPLITS = {
    "train": ["Pomaria_1_int", "Benevolence_2_int", "Beechwood_1_int", "Ihlen_0_int", "Benevolence_1_int", 
              "Pomaria_2_int", "Merom_1_int", "Ihlen_1_int", "Wainscott_0_int"],
    "val": ["Beechwood_0_int", "Wainscott_1_int", "Merom_0_int", "Rs_int", "Pomaria_0_int"]
}
# "Benevolence_0_int" in train


# SCENE_SPLITS = {
#     "train": ["Pomaria_1_int", "Benevolence_2_int"],
#     "val": ["Beechwood_0_int"]
# }


# for igibson
CATEGORIES = ['chair', 'table', 'picture', 'bottom_cabinet', 'cushion', 'sofa', 'bed', 'plant', 'sink', 'toilet', 'stool', 'standing_tv', 'shower', 'bathtub', 'counter']
# 15

CATEGORY_MAP = {
                'chair': 0,
                'table': 1,
                'picture': 2,
                'bottom_cabinet': 3,
                'cushion': 4,
                'sofa': 5,
                'bed': 6,
                'plant': 7,
                'sink': 8,
                'toilet': 9,
                'stool': 10,
                'standing_tv': 11,
                'shower': 12,
                'bathtub': 13,
                'counter': 14,
            }



# # for mp3d
# CATEGORIES = ['chair', 'table', 'picture', 'cabinet', 'cushion', 'sofa', 'bed', 'chest_of_drawers', 'plant', 'sink', 'toilet', 'stool', 'towel', 'tv_monitor', 'shower', 'bathtub', 'counter', 'fireplace', 'gym_equipment', 'seating', 'clothes']
# CATEGORY_MAP = {
#                 'chair': 3,
#                 'table': 5,
#                 'picture': 6,
#                 'cabinet': 7,
#                 'cushion': 8,
#                 'sofa': 10,
#                 'bed': 11,
#                 'chest_of_drawers': 13,
#                 'plant': 14,
#                 'sink': 15,
#                 'toilet': 18,
#                 'stool': 19,
#                 'towel': 20,
#                 'tv_monitor': 22,
#                 'shower': 23,
#                 'bathtub': 25,
#                 'counter': 26,
#                 'fireplace': 27,
#                 'gym_equipment': 33,
#                 'seating': 34,
#                 'clothes': 38
#             }



def initialize(num_processes):
    global scene_splits
    scene_splits = [[] for _ in range(num_processes)]
    for idx, scene in enumerate(SCENE_SPLITS['train']):
        scene_splits[idx % len(scene_splits)].append(scene)
    assert sum(map(len, scene_splits)) == len(SCENE_SPLITS['train'])
    
def getValue():
    global scene_splits
    return scene_splits

def getValValue():
    return SCENE_SPLITS['val']