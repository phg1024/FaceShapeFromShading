#ifndef FACESHAPEFROMSHADING_COMMON_H_H
#define FACESHAPEFROMSHADING_COMMON_H_H

#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <list>
#include <algorithm>
#include <functional>
#include <unordered_set>
#include <unordered_map>
#include <queue>
#include <limits.h>
using namespace std;

#ifndef MKL_BLAS
#define MKL_BLAS MKL_DOMAIN_BLAS
#endif

#define EIGEN_USE_MKL_ALL

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/LU>
#include <Eigen/Sparse>
#include <Eigen/CholmodSupport>
using namespace Eigen;

#include "glm/glm.hpp"
#include "gli/gli.hpp"

#endif //FACESHAPEFROMSHADING_COMMON_H_H
