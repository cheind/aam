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
#include <aam/pca.h>
#include <Eigen/Geometry>
#include <iostream>


TEST_CASE("pca-ray")
{
    // Sample 2d points from line model
    typedef Eigen::ParametrizedLine<float, 3> Ray;
    
    Ray r(Eigen::Vector3f(2, 3, 0),
          Eigen::Vector3f(1, 1, 1).normalized());

    Eigen::VectorXf ts = Eigen::VectorXf::Random(100);
    
    aam::MatrixX data(ts.size(), 3);
    for (Eigen::VectorXf::Index i = 0; i < ts.rows(); ++i) {
        data.row(i) = r.pointAt(ts(i));
    }
    
    // Compute PCA
    aam::RowVectorX mean(data.cols());
    aam::MatrixX basis(data.cols(), data.cols());
    aam::RowVectorX weights(data.cols());
    aam::computePCA(data, mean, basis, weights);
    
    Eigen::Vector3f::Index dims = aam::computePCADimensionality(weights, 0.00001f);
    REQUIRE(dims == 1);
    
    // Verify results
    REQUIRE(mean.isApprox(r.origin().transpose(), 0.1f));
    REQUIRE((basis.row(2) - r.direction().transpose()).norm() == Catch::Detail::Approx(0).epsilon(0.1));

    // Projection of data onto PCA basis is then a simple matter of matrix mul.
    Eigen::MatrixXf proj = data * basis.bottomRows(dims).transpose();
    const float corr = (r.origin().transpose() * basis.bottomRows(dims).transpose())(0);
    for (Eigen::VectorXf::Index i = 0; i < ts.rows(); ++i) {
        REQUIRE((proj(i, 0) - corr) == Catch::Detail::Approx(ts(i)).epsilon(0.1));
    }
}

TEST_CASE("pca-gaussian")
{
    aam::RowVector2 mean;
    mean << -1.f, 0.5f;
    aam::MatrixX cov = generate2DCovarianceMatrixFromStretchAndRotation(3, 0.01, 0.0);
    aam::MatrixX samples = sampleMultivariateGaussian(mean, cov, 50);

    aam::RowVectorX pcamean;
    aam::MatrixX pcabasis;
    aam::RowVectorX pcaweights;
    aam::computePCA(samples, pcamean, pcabasis, pcaweights);

    REQUIRE(pcamean.isApprox(mean, 0.1f));
    REQUIRE((pcabasis.row(1) - aam::RowVector2::Unit(0)).norm() == Catch::Detail::Approx(0).epsilon(0.1));
    REQUIRE((pcabasis.row(0) - aam::RowVector2::Unit(1)).norm() == Catch::Detail::Approx(0).epsilon(0.1));


    // Rotate by 45°
    cov = generate2DCovarianceMatrixFromStretchAndRotation(3, 0.01, 45.0 / 180.0 * 3.14159265359);
    samples = sampleMultivariateGaussian(mean, cov, 50);
    aam::computePCA(samples, pcamean, pcabasis, pcaweights);

    REQUIRE(pcamean.isApprox(mean, 0.1f));
    REQUIRE(std::abs(pcabasis.row(1).dot(aam::RowVector2(1, 1).normalized())) == Catch::Detail::Approx(1).epsilon(0.1));
    REQUIRE(std::abs(pcabasis.row(0).dot(aam::RowVector2(-1, 1).normalized())) == Catch::Detail::Approx(1).epsilon(0.1));
}