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

#ifndef AAM_DELAUNAY_H
#define AAM_DELAUNAY_H

#include <aam/types.h>

namespace aam {

    /** Rasterize shape.
     
        These method rasterizes the given triangle topology and produces a 
        sparse set of rasterized points represented as bary centric coordinates
        plus associated triangle id.
     
        \param normalizedShape List of normalized points in interleaved format x0, y0, x1, y1, ...
        \param triangleIds List of triangle vertices in triplets.
        \param imageWidth Width of image
        \param imageHeight Height of image
        \param shapeScale scaling to be applied to normalizedShape.
     
     */
    MatrixX rasterizeShape(
        Eigen::Ref<const RowVectorX> normalizedShape,
        Eigen::Ref<const RowVectorXi> triangleIds,
        MatrixX::Index imageWidth, MatrixX::Index imageHeight,
        Scalar shapeScale);

}

#endif