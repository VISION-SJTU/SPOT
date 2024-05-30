import os
import platform
import subprocess
from setuptools import Extension, dist, find_packages, setup

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

dist.Distribution().fetch_build_eggs(['Cython', 'numpy>=1.11.1'])
import numpy as np  # noqa: E402
from Cython.Build import cythonize  # noqa: E402


def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content


MAJOR = 1
MINOR = 0
PATCH = ''
SUFFIX = 'rc0'
SHORT_VERSION = '{}.{}.{}{}'.format(MAJOR, MINOR, PATCH, SUFFIX)


def get_version():
    return SHORT_VERSION


def make_cuda_ext(name, module, sources):

    return CUDAExtension(
        name='{}.{}'.format(module, name),
        sources=[os.path.join(*module.split('.'), p) for p in sources],
        extra_compile_args={
            'cxx': [],
            'nvcc': [
                '-D__CUDA_NO_HALF_OPERATORS__',
                '-D__CUDA_NO_HALF_CONVERSIONS__',
                '-D__CUDA_NO_HALF2_OPERATORS__',
            ]
        })


def make_cython_ext(name, module, sources):
    extra_compile_args = None
    if platform.system() != 'Windows':
        extra_compile_args = {
            'cxx': ['-Wno-unused-function', '-Wno-write-strings']
        }

    extension = Extension(
        '{}.{}'.format(module, name),
        [os.path.join(*module.split('.'), p) for p in sources],
        include_dirs=[np.get_include()],
        language='c++',
        extra_compile_args=extra_compile_args)
    extension, = cythonize(extension)
    return extension


if __name__ == '__main__':
    setup(
        name='spot',
        version=get_version(),
        description='SPOT: Sparsely-Supervised Object Tracking',
        long_description=readme(),
        author='VISION-SJTU',
        author_email='zhengjilai@sjtu.edu.cn',
        keywords='computer vision, object tracking',
        url='https://github.com/VISION-SJTU/SPOT/',
        packages=find_packages(exclude=('configs', 'tools', 'demo')),
        package_data={'spot.ops': ['*/*.so']},
        classifiers=[
            'Development Status :: 4 - Beta',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
        ],
        license='Apache License 2.0',
        setup_requires=['pytest-runner', 'cython', 'numpy'],
        tests_require=['pytest'],
        ext_modules=[
            make_cuda_ext(
                name='deform_conv_cuda',
                module='ltr.models.neck.dcn',
                sources=[
                    'src/deform_conv_cuda.cpp',
                    'src/deform_conv_cuda_kernel.cu'
                ]),
            make_cuda_ext(
                name='deform_pool_cuda',
                module='ltr.models.neck.dcn',
                sources=[
                    'src/deform_pool_cuda.cpp',
                    'src/deform_pool_cuda_kernel.cu'
                ]),
        ],
        cmdclass={'build_ext': BuildExtension},
        zip_safe=False)

    # Install pysot testing toolkit
    # os.chdir('./pysot_toolkit/toolkit/utils/')
    # os.system('python setup.py build_ext --inplace')
