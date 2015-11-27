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

#ifndef AAM_VIEWS_H
#define AAM_VIEWS_H

#include <aam/types.h>
#include <aam/traits.h>

namespace aam {

    template<class Scalar>
    inline typename AamMatrixTraits<Scalar>::MatrixMapType 
    toSeparatedView(typename AamMatrixTraits<Scalar, 1, Eigen::Dynamic>::MatrixRefType m, int dims = 2)
    {
        return AamMatrixTraits<Scalar>::MatrixMapType(m.data(), m.cols() / dims, dims, Eigen::Stride<Eigen::Dynamic, 1>(dims, 1));
    }

    template<class Scalar>
    inline typename AamMatrixTraits<Scalar>::ConstMatrixMapType
        toSeparatedViewConst(typename AamMatrixTraits<Scalar, 1, Eigen::Dynamic>::ConstMatrixRefType m, int dims = 2)
    {
        return AamMatrixTraits<Scalar>::ConstMatrixMapType(m.data(), m.cols() / dims, dims, Eigen::Stride<Eigen::Dynamic, 1>(dims, 1));
    }

    template<class Scalar>
    inline typename AamMatrixTraits<Scalar, 1, Eigen::Dynamic>::MatrixMapType
        toInterleavedView(typename AamMatrixTraits<Scalar>::MatrixRefType m)
    {
        return AamMatrixTraits<Scalar, 1, Eigen::Dynamic>::MatrixMapType(m.data(), 1, m.rows() * m.cols(), Eigen::Stride<Eigen::Dynamic, 1>(m.rows() * m.cols(), 1));
    }

    template<class Scalar>
    inline typename AamMatrixTraits<Scalar, 1, Eigen::Dynamic>::ConstMatrixMapType
        toInterleavedViewConst(typename AamMatrixTraits<Scalar>::ConstMatrixRefType m)
    {
        return AamMatrixTraits<Scalar, 1, Eigen::Dynamic>::ConstMatrixMapType(m.data(), 1, m.rows() * m.cols(), Eigen::Stride<Eigen::Dynamic, 1>(m.rows() * m.cols(), 1));
    }

}

#endif