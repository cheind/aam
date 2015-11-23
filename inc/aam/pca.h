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

#ifndef AAM_PCA_H
#define AAM_PCA_H

#include <aam/types.h>
#include <iostream>

namespace aam {
   
    /** Compute PCA transform for given data set.
     
        \param data MxN matrix with features in rows
        \param mean Mx1 matrix receiving the data mean
        \param basis NxK matrix with PCA normalized vectors in columns sorted by ascending eigenvalues.
     */
    void computePCA(Eigen::Ref<const MatrixX> data, Eigen::Ref<VectorX> mean, Eigen::Ref<MatrixX> basis, int k)
    {
        mean = data.colwise().mean();
        MatrixX centered = data.rowwise() - mean.transpose();
        MatrixX cov = centered.adjoint() * centered;

        Eigen::SelfAdjointEigenSolver<MatrixX> eig(cov);
        basis = eig.eigenvectors().rightCols(k);
    }

}

#endif