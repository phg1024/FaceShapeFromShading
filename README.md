# Face Shape from Shading
Yet another implementation of shape from shading for human face.

## Example
![Example](https://github.com/phg1024/FaceShapeFromShading/blob/master/bruce_model.jpg)

## Dependencies
* Boost 1.63
* freeglut
* GLEW
* glm
* gli
* SuiteSparse 4.5.3
* Eigen 3.3.3
* Intel MKL
* Qt5
* ceres solver 1.12.0
* Intel embree
* PhGLib

## Compile
```bash
git clone --recursive https://github.com/phg1024/FaceShapeFromShading.git
cd FaceShapeFromShading
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=icc -DCMAKE_CXX_COMPILER=icpc
make -j8
```
