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
    
    Eigen::MatrixXf data(ts.rows(), 3);
    for (Eigen::VectorXf::Index i = 0; i < ts.rows(); ++i) {
        data.row(i) = r.pointAt(ts(i));
    }
    
    // Compute PCA
    Eigen::Vector3f mean(data.cols());
    Eigen::MatrixXf basis(data.cols(), 1);
    aam::computePCA(data, mean, basis, 1); // Only interested in the most dominant basis direction.
    
    // Verify results
    REQUIRE(mean.isApprox(Eigen::Vector3f(2,3,0), 0.1f));
    REQUIRE((basis - r.direction()).norm() == Catch::Detail::Approx(0).epsilon(0.1f));

    
    
}
