import os
import json
import numpy as np
from IPython import embed
import shutil
from gibson2.utils.scene_converter import get_coords


def get_division():
    return {
        'chair': {
            'straight_chair': [
                '1b6c268811e1724ead75d368738e0b47',
                '2a8d87523e23a01d5f40874aec1ee3a6',
                '2db6c88145049555e6c5cd45aa112726',
                '2f42261a7f658407d12a1bc232f6622c',
                '3cc6485ab499244360b0434449421de1',
                '4c0983329afcd06f730e89ca0d2d13c3',
                '5a643c0c638fc2c3ff3a3ae710d23d1e',
                '8d4efa9893f7e49b3a85b56ecd01e269',
                '8d80cc5cdafc4b8929e17f7efc5a2421',
                '8d983a0adc059082b300c4ca2f51c01b',
                '8ec95f15623085a7b11ae648ea92233',
                '28a0b2a5afc96922ba63bc389be1ed5a',
                '97cd4ed02e022ce7174150bd56e389a8',
                '219c603c479be977d5e0096fb2d3266a',
                '239c363cbe7a650c8b1ad2cf16678177',
                '588bf81e78829fe7a16baf954c1d99bc',
                '640f61579181aef13ad3591a780fa12b',
                '662ecf4b0cd1f3d61f30b807ae39b61d',
                '791c14d53bd565f56ba14bfd91a75020',
                '7139284dff5142d4593ebeeedbff73b',
                '25957008f839ef647abe6643657b8aec',
                'b2ba1569509cdb439451566a8c6563ed',
                'cd5007a237ffde592b5bf1f191733d75',
                'cef1883847c02458cf44224546cb0306',
                'd0cf0982f16e5d583178d91f48c2217',
                'e07c7d5be62d7cd73ff4affcd321d45',
                'e8089df5cd0f9a573a3e9361d5a49edf',
                'ed108ed496777cf6490ad276cd2af3a4',
                'f2af2483f9fb980cb237f85c0ae7ac77',
            ],
            'arm_chair': [
                '4a329240c6a9d2547b11ae648ea92233',
                '2f1a67cdabe2a70c492d9da2668ec34c',
                '31a3884f500d9fa2025d98fb9de28cb',
                '35e8b034d84f60cb4d226702c1bfe9e2',
                '5107542cfbf142f36209799e55a657c',
                'c7f607892513a2f787bf0444104341d5',
                'c405857cd7d8aa04d225e12279334514',
                'd14d1da1e682a7cfaac8336231c53cd1',
                'f3f331e64d12b76d727e9f790cd597',
                'fc131dfba15fafb2fdeed357dfbe708a',
            ],
            'rocking_chair': [
                '4ce5a0d60fc942f2e595afbdc333be49',
                'a4da5746b99209f85da16758ae613576',
                'rocking_chair_0002',
            ],
            'chaise_longue': [
                '751d61e1d2cbbeaebdcc459b19e43a6',
                '758173c2c4630eab21f01d01c8c9dec6',
                'd324baae6630d7c8fb60456da917147',

            ],
            'folding_chair': [
                '100562',
                'folding_chair_0010',
                'folding_chair_0019',
            ],
            'highchair': [
                'dbfab57f9238e76799fc3b509229d3d',
                'high_chair',
            ]
        },
        'table': {
            'dining_table': [
                '1b4e6f9dd22a8c628ef9d976af675b86',
                '5f3f97d6854426cfb41eedea248a6d25',
                '9ad91992184e2b3e283b00891f680579',
                '33e4866b6db3f49e6fe3612af521500',
                '72c8fb162c90a716dc6d75c6559b82a2',
                '79f63a1564928af071a782a4379556c7',
                '98bab29db492767bc3bd24f986301745',
                '242b7dde571b99bd3002761e7a3ba3bd',
                '323ed7752b2a1db03ddaef18f97f546',
                '3344c70694e1bacdc5bd1ef3bc48a26',
                '19203',
                '26073',
                '26670',
                '265851637a59eb2f882f822c83877cbc',
                'b595da70965fff189427e63287029752',
                'c8cf1c77bbb79d214719088c8e42c6ab',
                'cafca523ae3653502454f22008de5a3e',
                'ccb96ea5f047c97f278d386bfa54545',
                'db665d85f1d9b1ea5c6a44a505804654',
                'e02925509615eb5a4eaf5bbf36d243d4',
                'fd958ba5f3116085492d9da2668ec34c',
            ],
            'desk': [
                '1f64fc95a612c646ecb720bdac052a97',
                '19d04a424a82894e641aac62064f7645',
                '783af15c06117bb29dd45a4e759f1d9c',
                '19898',
                '20043',
                '20745',
                '22339',
                '22367',
                '23372',
                '23511',
                '25913',
                '28164',
                '32052',
                '32746',
                '32932',
                '33457',
                'c356393b27c3fbca34ee3fb22432c207',
                'ea45801f26b84935d0ebb3b81115ac90'
            ],
            'pedestal_table': [
                '2fd962562b9f0370339797c21e8801b1',
                '10cc8c941fc8aeaa71a782a4379556c7',
                '22870',
                'bc48080ee5498d721fca2012865943e2',
                'f856245a7a9485deeb2d738c3fe5867f',
            ],
            'gaming_table': [
                '72a4fae0f304519dd8e0cfcf62e3e594',
            ],
            'stand': [
                '26525',
                '26652',
                '28668',
            ]
        },
    }


def divide_object_models():
    root_dir = '/cvgl2/u/chengshu/ig_dataset/objects'
    division = get_division()
    for category in division:
        obj_dir = os.path.join(root_dir, category)
        for subcat in division[category]:
            subcat_dir = os.path.join(root_dir, subcat)
            os.makedirs(subcat_dir, exist_ok=True)
            for obj_inst in division[category][subcat]:
                src = os.path.join(obj_dir, obj_inst)
                dst = os.path.join(subcat_dir, obj_inst)
                assert os.path.isdir(src), '{} does not exist'.format(src)
                if os.path.isdir(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)

        # Uncomment this if you want to remove the original category
        # shutil.rmtree(obj_dir)


def main():
    divide_object_models()


if __name__ == "__main__":
    main()
