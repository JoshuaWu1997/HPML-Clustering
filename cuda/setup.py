from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='my_cuda_util',
      ext_modules=[cpp_extension.CUDAExtension('my_cuda_util', ['my_cuda_util.cpp', 'my_cuda_kernel.cu'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
