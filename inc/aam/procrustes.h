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

#ifndef AAM_PROCRUSTES_H
#define AAM_PROCRUSTES_H

#include <aam/types.h>

namespace aam {
   
    /** Compute Procrustes shape normalization.

        Procrustes analysis determines a linear transformation (translation,
        reflection, orthogonal rotation and scaling) of the points in Y to best
        conform them to the points in matrix X, using the sum of squared errors
        as the goodness of fit criterion.
     
        \param X Nx2 Target shape consisting of N two-dimensional measurements.
        \param Y Nx2 Input shape consisting of N two-dimensional measurements.
        \param t 3x3 Similarity transform applied to Y.
        \return Normalized distance between X and transformed Y.
     */
    Scalar procrustes(Eigen::Ref<const MatrixX> X, Eigen::Ref<MatrixX> Y);

}

#endif