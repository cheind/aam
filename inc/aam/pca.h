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
     
        \param data MxN matrix with N features in M dimensions in columns
        \param mean Mx1 matrix receiving the data mean
        \param basis MxM matrix with PCA normalized vectors in columns sorted by ascending eigenvalues.
        \param weights Mx1 matrix containing the eigenvalues sorted in ascending order.
     */
    void computePCA(Eigen::Ref<const MatrixX> data, Eigen::Ref<VectorX> mean, Eigen::Ref<MatrixX> basis, Eigen::Ref<VectorX> weights)
    {
        mean = data.rowwise().mean();
        MatrixX centered = data.colwise() - mean;
        MatrixX cov = centered * centered.adjoint();

        Eigen::SelfAdjointEigenSolver<MatrixX> eig(cov);
        basis = eig.eigenvectors();
        weights = eig.eigenvalues();
    }
    
    /** Compute the PCA subspace dimensionality for a given tolerated loss.
     
        \param weights eigen values sorted in ascending order
        \param toleratedCompressionLoss allowed compression loss in percent
        \return Returns the number of dimensions subspace.
     
     */
    inline VectorX::Index computePCADimensionality(Eigen::Ref<const VectorX> weights, MatrixX::Scalar toleratedCompressionLoss) {
        VectorX::Scalar sum = weights.sum();
        VectorX::Scalar loss = 0.f;
        VectorX::Index idx = 0;
        
        while (loss <= toleratedCompressionLoss && idx < weights.cols()) {
            loss += weights(idx) / sum;
            idx++;
        }
        
        return weights.cols() - idx + 1;
    }

}

#endif