/**
This file is part of Active Appearance Models (AAM).

Copyright Christoph Heindl 2015
Copyright Sebastian Zambal 2015

AAM is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

AAM is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with AAM.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef AAM_TRAITS_H
#define AAM_TRAITS_H

#include <Eigen/Core>

namespace aam {
   
    /** Default scalar to matrix traits. */
    template<class Scalar, int rows = Eigen::Dynamic, int cols = Eigen::Dynamic>
    struct AamMatrixTraits {
        typedef Eigen::Matrix<Scalar, rows, cols, Eigen::RowMajor> MatrixType;
        typedef Eigen::Map<MatrixType, 0, Eigen::Stride<Eigen::Dynamic, 1> > MatrixMapType;
        typedef Eigen::Ref<MatrixType> MatrixRefType;
        typedef Eigen::Ref<MatrixType const> ConstMatrixRefType;
    };
 

}

#endif