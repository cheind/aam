/**
This file is part of Active Appearance Models (AMM).

Copyright Christoph Heindl 2015
Copyright Sebastian Zambal 2015

AMM is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

AMM is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with AMM.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef AAM_RANDOM_DISTRIBUTIONS_H
#define AAM_RANDOM_DISTRIBUTIONS_H

#include <aam/types.h>
#include <Eigen/Dense>
#include <random>

namespace Eigen {
    namespace internal {

        template<class Scalar>
        struct ScalarNormalDistOp
        {
            static std::mt19937 rng;
            mutable std::normal_distribution<Scalar> norm;

            EIGEN_EMPTY_STRUCT_CTOR(ScalarNormalDistOp)

            template<typename Index>
            inline const Scalar operator() (Index, Index = 0) const
            { return norm(rng); }
        };
        template<class Scalar>
        std::mt19937 ScalarNormalDistOp<Scalar>::rng;

        template<class Scalar>
        struct functor_traits< ScalarNormalDistOp<Scalar> >
        {
            enum { Cost = 50 * NumTraits<aam::Scalar>::MulCost, PacketAccess = false, IsRepeatable = false };
        };

    } 
}

inline aam::MatrixX sampleMultivariateGaussian(Eigen::Ref<const aam::VectorX> mean, Eigen::Ref<const aam::MatrixX> covar, int samples, unsigned long seed = 5489UL)
{
    Eigen::SelfAdjointEigenSolver<aam::MatrixX> solver(covar);
    aam::MatrixX transform  = solver.eigenvectors() * solver.eigenvalues().cwiseMax(0).cwiseSqrt().asDiagonal();
    Eigen::internal::ScalarNormalDistOp<aam::Scalar> op;
    op.rng.seed(seed);
    return (transform * aam::MatrixX::NullaryExpr(mean.rows(), samples, op)).colwise() + mean;
}

inline aam::MatrixX generate2DCovarianceMatrixFromStretchAndRotation(double varX, double varY, double theta)
{
    aam::MatrixX rot = Eigen::Rotation2D<double>(theta).matrix().cast<aam::Scalar>();
    return rot*Eigen::DiagonalMatrix<aam::Scalar, 2, 2>(aam::Scalar(varX), aam::Scalar(varY))*rot.transpose();
}

#endif