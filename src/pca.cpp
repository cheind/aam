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
#include <iostream>

namespace aam {

    void computePCA(Eigen::Ref<const MatrixX> data, RowVectorX &mean, MatrixX &basis, RowVectorX &weights)
    {
        mean = data.colwise().mean();
        MatrixX centered = data.rowwise() - mean;

        if (data.rows() > data.cols()) {  // use regular Eigen-analysis

            MatrixX cov = (centered.adjoint() * centered) * ((aam::Scalar)1.0 / (aam::Scalar)(data.rows() - 1));

            Eigen::SelfAdjointEigenSolver<MatrixX> eig(cov);
            basis = eig.eigenvectors().transpose();
            weights = eig.eigenvalues();

        } else {  // use "trick" (see http://www.doc.ic.ac.uk/~dfg/ProbabilisticInference/IDAPILecture15.pdf, Page 5) 
                  // to reduce size of covariance matrix. Important: need to multiply by centered.transpose() and 
                  // apply normalization of eigenvectors afterwards!

            MatrixX cov2 = (centered * centered.adjoint()) * ((aam::Scalar)1.0 / (aam::Scalar)(data.rows() - 1));

            Eigen::SelfAdjointEigenSolver<MatrixX> eig2(cov2);
            MatrixX b = (centered.transpose() * eig2.eigenvectors()).transpose();
            MatrixX n = b.rowwise().norm();
            b.rowwise().normalize();    
            basis = b;
            weights = eig2.eigenvalues();
        }
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