---
layout: post
title: Install Caffe on Fedora 21 without Cuda
date: 2014-04-21 16:39
categories: 
    - deep learning
tags:
    - caffe
comments: true

---
[Caffe](https://github.com/BVLC/caffe) is a popular deep learning framework. 
Installing [caffe](https://github.com/BVLC/caffe) on fedora 21 is not very easy. 
One reason is that following the [official install instructions](http://caffe.berkeleyvision.org/installation.html) can't always make you successful.
Another reason is that fedora seems to be less popular than Ubuntu and there are few reports on how to solve some strange problems. 
During installation, I met with many problems. This post records these problems on X86\_64 fedora platform.  

[Cuda 7](https://developer.nvidia.com/cuda-downloads) is still not available on fedora 21. 
Cuda 6.5 can't be complied using the default gcc complier. 
Therefore, I don't install cuda and only use the cpu mode of caffe.   
##Basic Installation
First, follow the [fedora installing instruction](http://caffe.berkeleyvision.org/install_yum.html) to install dependency packages. 
##Problem 1
Then, enter the caffe directory and type ``cmake .``, You may be see the error below:
<pre>
 Could NOT find Atlas (missing: Atlas_LAPACK_LIBRARY)
 Call Stack (most recent call first):
 /usr/share/cmake/Modules/FindPackageHandleStandardArgs.cmake:343 (_FPHSA_FAILURE_MESSAGE)
       cmake/Modules/FindAtlas.cmake:43 (find_package_handle_standard_args)
          cmake/Dependencies.cmake:74 (find_package)
            CMakeLists.txt:26 (include)
</pre>
**Solution**  
In file **cmake/Modules/FindAtlas.cmake** line 31, we need to change  
<pre>
find_library(Atlas_LAPACK_LIBRARY NAMES alapack_r alapack lapack_atlas PATHS ${Atlas_LIB_SEARCH_PATHS})
</pre>
to  
<pre>
find_library(Atlas_LAPACK_LIBRARY NAMES alapack_r alapack lapack_atlas lapack PATHS ${Atlas_LIB_SEARCH_PATHS})
</pre>
We add **lapack** in find\_library.  
##Using openblas
<pre>
cp Makefile.config.example Makefile.config
</pre>
Modify Makefile.config:
1. Comment Out Line 8.   
Change  
<pre>
\#CPU_ONLY=1
</pre>
to  
<pre>
CPU_ONLY=1
</pre>
2. Comment Out Line 15 to disable cuda.  
3. In Line 33, set BLAS to be open. (or you can use atlas. ) 
<pre>
BLAS:= open
</pre>
4. In Line 65, add **/usr/lib64/** to **LIBRARY\_DIRS**.   
<pre>
LIBRARY_DIRS := $(PYTHON\_LIB) /usr/local/lib /usr/lib /usr/lib64
</pre>
5. make
<pre>
make all
</pre>
##Problem 2
Maybe you will meet these problems.
<pre>
../lib/libcaffe.so: undefined reference to `cblas_sgemv'
../lib/libcaffe.so: undefined reference to `cblas_dgemm'
../lib/libcaffe.so: undefined reference to `cblas_sscal'
../lib/libcaffe.so: undefined reference to `cblas_dgemv'
../lib/libcaffe.so: undefined reference to `cblas_saxpy'
../lib/libcaffe.so: undefined reference to `cblas_ddot'
../lib/libcaffe.so: undefined reference to `cblas_dasum'
../lib/libcaffe.so: undefined reference to `cblas_sgemm'
../lib/libcaffe.so: undefined reference to `cblas_dscal'
../lib/libcaffe.so: undefined reference to `cblas_scopy'
../lib/libcaffe.so: undefined reference to `cblas_sasum'
../lib/libcaffe.so: undefined reference to `cblas_daxpy'
../lib/libcaffe.so: undefined reference to `cblas_dcopy'
../lib/libcaffe.so: undefined reference to `cblas_sdot'
</pre>
This error means we are missing some libraries.   
The solution is to add **/usr/lib64/openblas64.so** below proto into src/caffe/CMakeLists.txt.  
<pre>
target_link_libraries(caffe proto ${Caffe_LINKER_LIBS})
target_link_libraries(caffe /usr/lib64/libopenblas64.so)
</pre>
Then,
<pre>
cmake .
make all
make test
make runtest
</pre>
