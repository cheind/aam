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

#include <aam/pca.h>
#include <Eigen/Geometry>
#include <iostream>


TEST_CASE("pca")
{
    // Sample 2d points from line model
    typedef Eigen::ParametrizedLine<float, 3> Ray;
    
    Ray r(Eigen::Vector3f(2, 3, 0),
          Eigen::Vector3f(1, 1, 1).normalized());

    Eigen::VectorXf ts = Eigen::VectorXf::Random(1000);
    
    Eigen::MatrixXf data(3, ts.size());
    for (Eigen::VectorXf::Index i = 0; i < ts.rows(); ++i) {
        data.col(i) = r.pointAt(ts(i));
    }
    
    // Compute PCA
    Eigen::Vector3f mean(data.rows());
    Eigen::MatrixXf basis(data.rows(), data.rows());
    aam::computePCA(data, mean, basis); // Only interested in the most dominant basis direction.
    
    // Verify results
    REQUIRE(mean.isApprox(Eigen::Vector3f(2,3,0), 0.1f));
    REQUIRE((basis.col(2) - r.direction()).norm() == Catch::Detail::Approx(0).epsilon(0.1));

    // Projection of data onto PCA basis is then a simple matter of matrix mul.
    Eigen::MatrixXf proj = basis.rightCols(1).transpose() * data;
    const float corr = (basis.rightCols(1).transpose() * Eigen::Vector3f(2, 3, 0))(0);
    for (Eigen::VectorXf::Index i = 0; i < ts.rows(); ++i) {
        REQUIRE((proj(0, i) - corr) == Catch::Detail::Approx(ts(i)).epsilon(0.1));
    }
    
}
