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


#include <aam/procrustes.h>
#include <aam/map.h>
#include <Eigen/Dense>
#include <iostream>

namespace aam {
    
    Scalar procrustes(Eigen::Ref<const MatrixX> X, Eigen::Ref<MatrixX> Y)
    {
        RowVector2 meanX = X.colwise().mean();
        RowVector2 meanY = Y.colwise().mean();

        MatrixX centeredX = X.rowwise() - meanX;
        MatrixX centeredY = Y.rowwise() - meanY;

        // Compute Frobenius norm. 
        const Scalar sX = centeredX.norm();
        const Scalar sY = centeredY.norm();

        // Scale to unit norm
        centeredX /= sX;
        centeredY /= sY;

        // Find optimal rotation based on correlation of landmarks
        MatrixX A = centeredX.transpose() * centeredY;
        auto svd = A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);

        // Equation 7.
        Matrix2 v = svd.matrixV();
        Matrix2 u = svd.matrixU();
        RowVector2 s = svd.singularValues();
        Matrix2 rot = v * u.transpose();

        // Make sure we don't suffer from reflection.
        if (rot.determinant() < 0) {
            v.rightCols(1) *= -1;
            s.rightCols(1) *= -1;
            rot = v * u.transpose();
        }

        Scalar trace = s.sum();

        // Scaling of Y
        // Scalar b = trace * sX / sY;

        // Distance of X and T(Y)
        Scalar d = 1 - trace * trace;

        // Transform Y
        Y = (((centeredY * rot) * trace * sX).rowwise() + meanX).eval();

        return d;
    }

    Scalar generalizedProcrustes(Eigen::Ref<MatrixX> X, int maxIterations)
    {
        MatrixX shapes = fromInterleaved<Scalar>(X);        
        MatrixX::Index nShapes = X.rows();
        MatrixX::Index rowsPerShape = shapes.rows();

        // Perform iterative optimization
        // - arbitrarily choose a reference shape(typically by selecting it among the available instances)
        // - superimpose all instances to current reference shape
        // - compute the mean shape of the current set of superimposed shapes
        // - if the Procrustes distance between mean and reference shape is above a threshold, set reference to mean shape and continue to step 2.
        
        bool done = false;
        MatrixX refShape = shapes.block<Eigen::Dynamic, 2>(0, 0, rowsPerShape, 2);
        Scalar lastDist = std::numeric_limits<Scalar>::max();
        int iterations = 0;
        do {
            for (MatrixX::Index s = 0; s < nShapes; ++s) {
                procrustes(refShape, shapes.block<Eigen::Dynamic, 2>(0, s * 2, rowsPerShape, 2));
            }

            MatrixX meanShape = MatrixX::Zero(rowsPerShape, 2);
            for (MatrixX::Index s = 0; s < nShapes; ++s) {
                meanShape += shapes.block<Eigen::Dynamic, 2>(0, s * 2, rowsPerShape, 2);
            }
            meanShape /= (Scalar)nShapes;

            Scalar dist = (meanShape - refShape).norm();
            if (dist > lastDist || ++iterations > maxIterations)
                done = true;


            lastDist = dist;
            refShape = meanShape;

        }  while (!done);

        // Write back as interleaved
        X = toInterleaved<Scalar>(shapes);

        return lastDist;
    }
}
