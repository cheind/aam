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

#include "catch.hpp"
#include "random_distributions.h"
#include <aam/procrustes.h>
#include <Eigen/Geometry>
#include <iostream>


TEST_CASE("procrustes")
{
    aam::RowVector2 mean;
    mean << -2.f, -2.f;
    aam::MatrixX cov = generate2DCovarianceMatrixFromStretchAndRotation(3, 3, 0.0);
    
    aam::MatrixX X = sampleMultivariateGaussian(mean, cov, 5);

    // Note, Transform assumes column vectors, so we need to transpose it when applied to a matrix.
    Eigen::Transform<aam::Scalar, 2, Eigen::Affine> sim;
    sim = Eigen::Rotation2D<aam::Scalar>(3.1415f) * Eigen::Scaling(5.f) * Eigen::Translation<aam::Scalar, 2>(-10.f, -10.f);

    aam::MatrixX Y = (X.rowwise().homogeneous() * sim.matrix().transpose()).rowwise().hnormalized();

    REQUIRE(!X.isApprox(Y));
    REQUIRE(aam::procrustes(X, Y) == Approx(0).epsilon(0.1));
    REQUIRE(X.isApprox(Y));
}
