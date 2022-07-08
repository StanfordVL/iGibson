class dataset:
    def __init__(self, scene):
        self.SCENE_SPLITS = None
        self.scene_splits = None
        
        if scene == 'igibson':
            self.SCENE_SPLITS = {
                "train": ["Pomaria_1_int", "Benevolence_2_int", "Beechwood_1_int", "Ihlen_0_int", "Benevolence_1_int", 
                          "Pomaria_2_int", "Merom_1_int", "Ihlen_1_int", "Wainscott_0_int"],  # "Benevolence_0_int"], 
                "val": ["Beechwood_0_int", "Wainscott_1_int", "Merom_0_int", "Rs_int", "Pomaria_0_int"]
            }
        if scene == 'mp3d' or scene == "gibson":
            self.SCENE_SPLITS = {
                'train': ['sT4fr6TAbpF', 'E9uDoFAP3SH', 'VzqfbhrpDEA', 'kEZ7cmS4wCh', '29hnd4uzFmX', 'ac26ZMwG7aT',
                          'i5noydFURQK', 's8pcmisQ38h', 'rPc6DW4iMge', 'EDJbREhghzL', 'mJXqzFtmKg4', 'B6ByNegPMKs',
                          'JeFG25nYj2p', '82sE5b5pLXE', 'D7N2EKCX4Sj', '7y3sRwLe3Va', 'HxpKQynjfin', '5LpN3gDmAk7',
                          'gTV8FGcVJC9', 'ur6pFq6Qu1A', 'qoiz87JEwZ2', 'PuKPg4mmafe', 'VLzqgDo317F', 'aayBHfsNo7d',
                          'JmbYfDe2QKZ', 'XcA2TqTSSAj', '8WUmhLawc2A', 'sKLMLpTHeUy', 'r47D5H71a5s', 'Uxmj2M2itWa',
                          'Pm6F8kyY3z2', 'p5wJjkQkbXX', '759xd9YjKW5', 'JF19kD82Mey', 'V2XKFyX4ASd', '1LXtFkjw3qL',
                          '17DRP5sb8fy', '5q7pvUzZiYa', 'VVfe2KiqLaN', 'Vvot9Ly1tCj', 'ULsKaCPVFJR', 'D7G3Y4RVNrH',
                          'uNb9QFRL6hY', 'ZMojNkEp431', '2n8kARJN3HM', 'vyrNrziPKCB', 'e9zR4mvMWw7', 'r1Q1Z4BcV1o',
                          'PX4nDJXEHrG', 'YmJkqBEsHnH', 'b8cTxDM8gDG', 'GdvgFV5R1Z5', 'pRbA3pwrgk9', 'jh4fc5c5qoQ',
                          '1pXnuDYAj8r', 'S9hNv5qa7GM', 'VFuaQ6m2Qom', 'cV4RVeZvu5T', 'SN83YJsR3w2'],
                'val': ['x8F5xyUWy9e', 'QUCTc6BB5sX', 'EU6Fwq7SyZv', '2azQ1b91cZZ', 'Z6MFQCViBuw', 'pLe4wQe7qrG',
                        'oLBMNvg9in8', 'X7HyMhZNoso', 'zsNo4HB9uLZ', 'TbHJrupSAjP', '8194nk5LbLH', 'pa4otMbVnkk', 
                        'yqstnuAEVhm', '5ZKStnWn8Zo', 'Vt2qJdWjCF2', 'wc2JMjhGNzB', 'WYY7iVyf5p8',
                         'fzynW3qQPVF', 'UwV83HsGsw3', 'q9vSo1VnCiC', 'ARNzJeq3xxb', 'rqfALeAoiTq', 'gYvKGZ5eRqb',
                         'YFuZgdQ5vWj', 'jtcxE69GiFV', 'gxdoqLR6rwA'],
            #     'test': ['pa4otMbVnkk', 'yqstnuAEVhm', '5ZKStnWn8Zo', 'Vt2qJdWjCF2', 'wc2JMjhGNzB', 'WYY7iVyf5p8',
            #              'fzynW3qQPVF', 'UwV83HsGsw3', 'q9vSo1VnCiC', 'ARNzJeq3xxb', 'rqfALeAoiTq', 'gYvKGZ5eRqb',
            #              'YFuZgdQ5vWj', 'jtcxE69GiFV', 'gxdoqLR6rwA'],
            }
            
        
    def split(self, num_processes):
        self.scene_splits = [[] for _ in range(num_processes)]
        for idx, scene in enumerate(self.SCENE_SPLITS['train']):
            self.scene_splits[idx % len(self.scene_splits)].append(scene)
        assert sum(map(len, self.scene_splits)) == len(self.SCENE_SPLITS['train'])
        return self.scene_splits