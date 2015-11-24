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


#include <aam/pca.h>
#include <Eigen/Dense>

namespace aam {
    
    void computePCA(Eigen::Ref<const MatrixX> data, Eigen::Ref<RowVectorX> mean, Eigen::Ref<MatrixX> basis, Eigen::Ref<RowVectorX> weights)
    {
        mean = data.colwise().mean();
        MatrixX centered = data.rowwise() - mean;
        MatrixX cov = centered.adjoint() * centered;

        Eigen::SelfAdjointEigenSolver<MatrixX> eig(cov);
        basis = eig.eigenvectors().transpose();
        weights = eig.eigenvalues();
    }

    RowVectorX::Index computePCADimensionality(Eigen::Ref<const RowVectorX> weights, MatrixX::Scalar toleratedCompressionLoss) {
        RowVectorX::Scalar sum = weights.sum();
        RowVectorX::Scalar loss = 0.f;
        RowVectorX::Index idx = 0;

        while (loss <= toleratedCompressionLoss && idx <= weights.cols()) {
            loss += weights(idx) / sum;
            idx++;
        }

        return weights.cols() - idx + 1;
    }
}