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


#include <aam/transform.h>
#include <aam/views.h>
#include <Eigen/Dense>

namespace aam {

    
    void transformShape(const Affine2 &t, Eigen::Ref<const RowVectorX> src, Eigen::Ref<RowVectorX> dst)
    {
        dst = toInterleavedViewConst<Scalar>(toSeparatedViewConst<Scalar>(src).rowwise().homogeneous() * t);
    }
    
    void transformShapeInPlace(const Affine2 &t, Eigen::Ref<RowVectorX> srcdst)
    {
        auto x = toSeparatedView<Scalar>(srcdst);
        x = (x.rowwise().homogeneous() * t).eval();
    }

    RowVectorX transformShape(const Affine2 &t, Eigen::Ref<const RowVectorX> src)
    {
        return toInterleavedViewConst<Scalar>(toSeparatedViewConst<Scalar>(src).rowwise().homogeneous() * t);
    }
}