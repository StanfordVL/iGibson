from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install
from subprocess import check_call
from distutils.command.build_py import build_py as _build_py
import sys, os.path
import re

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion
import subprocess
import platform
import codecs
import platform
import shutil


use_clang = False

here = os.path.abspath(os.path.dirname(__file__))

def read(*parts):
    with codecs.open(os.path.join(here, *parts), 'r') as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)', out.decode()).group(1))
            if cmake_version < '3.1.0':
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

        if platform.system() == "Windows":
            mesh_renderer_dir = os.path.join(here, 'gibson2', 'render', 'mesh_renderer')
            release_dir = os.path.join(mesh_renderer_dir, 'Release')
            for f in os.listdir(release_dir):
                shutil.copy(os.path.join(release_dir, f), mesh_renderer_dir)

            shutil.rmtree(release_dir)
            vr_dll = os.path.join(here, 'gibson2', 'render', 'openvr', 'bin', 'win64', 'openvr_api.dll')
            sr_ani_dir = os.path.join(here, 'gibson2', 'render', 'sranipal', 'bin')
            shutil.copy(vr_dll, mesh_renderer_dir)

            for f in os.listdir(sr_ani_dir):
                if f.endswith('dll'):
                    shutil.copy(os.path.join(sr_ani_dir, f), mesh_renderer_dir)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' +
            os.path.join(extdir, 'gibson2', 'render', 'mesh_renderer'),
            '-DCMAKE_RUNTIME_OUTPUT_DIRECTORY=' +
            os.path.join(extdir, 'gibson2', 'render', 'mesh_renderer', 'build'),
            '-DPYTHON_EXECUTABLE=' + sys.executable
        ]

        if use_clang:
            cmake_args += [
                '-DCMAKE_C_COMPILER=/usr/bin/clang', '-DCMAKE_CXX_COMPILER=/usr/bin/clang++'
            ]

        if platform.system() == 'Darwin':
            cmake_args += ['-DMAC_PLATFORM=TRUE']
        else:
            cmake_args += ['-DMAC_PLATFORM=FALSE']

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY{}={}'.format(cfg.upper(), extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j2']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                              self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='gibson2',
    version='0.0.5',
    author='Stanford University',
    long_description_content_type="text/markdown",
    long_description=long_description,
    url='https://github.com/StanfordVL/iGibson',
    zip_safe=False,
    packages=find_packages(),
    install_requires=[
            'gym>=0.12',
            'numpy>=1.16.0',
            'scipy>=1.2.1',
            'pybullet>2.6.4',
            'transforms3d>=0.3.1',
            'opencv-python>=3.4.8',
            'Pillow>=5.4.0',
            'networkx>=2.0',
            'PyYAML',
            'tqdm',
            'matplotlib',
            'cloudpickle',
            'aenum',
            'GPUtil',
            'ipython',
            'pytest',
            'future',
            'trimesh',
            'sphinx_markdown_tables',
            'sphinx==2.2.1',
            'recommonmark',
            'sphinx_rtd_theme'
    ],
    ext_modules=[CMakeExtension('MeshRendererContext', sourcedir='gibson2/render')],
    cmdclass=dict(build_ext=CMakeBuild),
    tests_require=[],
    package_data={'': [
    'gibson2/global_config.yaml',
    'gibson2/render/mesh_renderer/shaders/*'
    ]},
    include_package_data=True,
)   #yapf: disable
