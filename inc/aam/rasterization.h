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
        \return Nx3 matrix containing a triplet of triangleId, alpha, beta per row.
     */
    MatrixX rasterizeShape(Eigen::Ref<const RowVectorX> normalizedShape,
                           Eigen::Ref<const RowVectorXi> triangleIds,
                           MatrixX::Index imageWidth, MatrixX::Index imageHeight,
                           Scalar shapeScale);

    /** Generate image from shape and sparse set of rasterization positions.
     
        \param normalizedShape List of normalized points in interleaved format x0, y0, x1, y1, ...
        \param triangleIds List of triangle vertices in triplets.
        \param barycentricSamplePositions Nx3 matrix containing triplets of triangleId, alpha, beta per row.
        \param colorsAtSamplePositions NxM matrix of color values per sample position. The number of channels in resulting
               image is M.
        \param backgroundColor 1xM row vector containing the background color of the image.
        \param image Output image matrix, pre-allocated.
        \param shapeScale scaling to be applied to normalizedShape.
     
     */
    void generateImageFromRasterizedPositions(Eigen::Ref<const RowVectorX> normalizedShape,
                           Eigen::Ref<const RowVectorXi> triangleIds,
                           Eigen::Ref<const MatrixX> barycentricSamplePositions,
                           Eigen::Ref<const MatrixX> colorsAtSamplePositions,
                           Eigen::Ref<const RowVectorX> backgroundColor,
                           Scalar shapeScale,
                           Eigen::Ref<MatrixX> image);
    
    
}

#endif