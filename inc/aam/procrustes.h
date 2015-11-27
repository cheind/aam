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
     
        References:
        Stegmann, Mikkel B., and David Delgado Gomez. 
        "A brief introduction to statistical shape analysis." 
        Informatics and Mathematical Modelling, Technical University of Denmark, DTU 15 (2002): 11.
     
        \param X Nx2 Target shape consisting of N two-dimensional measurements.
        \param Y Nx2 Input shape consisting of N two-dimensional measurements. Modified in place.
        \return Normalized distance between X and transformed Y.
     */
    Scalar procrustes(Eigen::Ref<const RowVectorX> X, Eigen::Ref<RowVectorX> Y);

    /** Compute Procrustes shape normalization of n-shapes.

        Similar to aam::procrustes but computes normalizations for a set of shapes concurrently.
        Optimization is based on iterative approach that compares a reference shape to the current
        mean shape in each iteration. If the Procrustes distance is above a threshold, the mean
        shape is made the new reference mesh and the binary procrustes is applied between the new
        reference mesh and each shape.
     
        Based on
        Stegmann, Mikkel B., and David Delgado Gomez.
        "A brief introduction to statistical shape analysis."
        Informatics and Mathematical Modelling, Technical University of Denmark, DTU 15 (2002): 11.

        \param X NxM Shape Matrix
        \param maxIteraions Maximum number of iterations to perform normalization.
        \return Normalized distance between mean shape and reference shape in last iteration.
    */
    MatrixX generalizedProcrustes(Eigen::Ref<const MatrixX> X, int maxIterations);

}

#endif