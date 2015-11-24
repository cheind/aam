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
#include "test_config.h"
#include <aam/procrustes.h>
#include <aam/show.h>
#include <aam/map.h>
#include <Eigen/Geometry>
#include <iostream>
#include <opencv2/opencv.hpp>


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

TEST_CASE("procrustes2") {
    aam::MatrixX X(2, 116);
    X.row(0) << 0.35764, 0.64078, 0.36268, 0.68243, 0.37477, 0.72946, 0.39897, 0.78052, 0.42215, 0.81680, 0.45945, 0.84770, 0.49070, 0.85442, 0.53707, 0.83964,             0.56429, 0.81008, 0.58243, 0.77246, 0.59756, 0.72812, 0.60764, 0.69049, 0.61469, 0.65019, 0.58445, 0.50776, 0.56933, 0.49298, 0.55320, 0.48761, 0.53102, 0.49029, 0.52296, 0.50642, 0.54110, 0.51179, 0.55622, 0.51582, 0.57235, 0.51582, 0.39292, 0.50507, 0.41106, 0.48895, 0.43022, 0.48089, 0.44836, 0.49164, 0.46449, 0.50507, 0.44635, 0.51045, 0.42820, 0.51179, 0.41006, 0.50910, 0.50784, 0.47551, 0.52094, 0.44864, 0.57336, 0.43789, 0.60562, 0.45670, 0.61167, 0.50239, 0.47659, 0.47283, 0.45743, 0.44461, 0.40502, 0.43386, 0.37578, 0.44864, 0.36973, 0.49298, 0.43425, 0.69453, 0.47659, 0.68646, 0.48969, 0.69453, 0.50280, 0.68512, 0.54514, 0.68646, 0.50885, 0.72409, 0.49070, 0.72543, 0.47256, 0.72409, 0.47457, 0.50910, 0.47558, 0.56016, 0.45340, 0.59241, 0.45139, 0.62466, 0.46852, 0.63272, 0.49070, 0.63137, 0.51389, 0.63272, 0.52901, 0.61525, 0.52094, 0.58569, 0.50885, 0.56151, 0.50683, 0.51179;
    
    X.row(1) << 0.41106, 0.67571, 0.41409, 0.70662, 0.42215, 0.74424, 0.43727, 0.77380, 0.45945, 0.80605, 0.49171, 0.83695, 0.53405, 0.84098, 0.57437, 0.83829, 0.61167, 0.80605, 0.62578, 0.77380, 0.64090, 0.74693, 0.64897, 0.71065, 0.65300, 0.67303, 0.62377, 0.55613, 0.60864, 0.54673, 0.59352, 0.54135, 0.57840, 0.54673, 0.56227, 0.55613, 0.57840, 0.56016, 0.59453, 0.56688, 0.61167, 0.56554, 0.44836, 0.55210, 0.45743, 0.54404, 0.47256, 0.53732, 0.48768, 0.54538, 0.50381, 0.55613, 0.48868, 0.56419, 0.47256, 0.56957, 0.45844, 0.56151, 0.56933, 0.50373, 0.59352, 0.49567, 0.61066, 0.49432, 0.62881, 0.50507, 0.64292, 0.52791, 0.49574, 0.50104, 0.47356, 0.49567, 0.44635, 0.49029, 0.43828, 0.49835, 0.42518, 0.51985, 0.48163, 0.71871, 0.52195, 0.70796, 0.53808, 0.71065, 0.55118, 0.70527, 0.58949, 0.71199, 0.55522, 0.74290, 0.53506, 0.74424, 0.51893, 0.74290, 0.51792, 0.56419, 0.51792, 0.60181, 0.49977, 0.62869, 0.49776, 0.65153, 0.50481, 0.66228, 0.53707, 0.66765, 0.56429, 0.65959, 0.57235, 0.64347, 0.56731, 0.62466, 0.55723, 0.60181, 0.55320, 0.56285;
    
#ifdef AAM_TESTS_VERBOSE
    cv::Mat shapes = aam::toOpenCVHeader(X);
    cv::Mat img(480, 640, CV_8UC3);
    img.setTo(0);
    aam::drawShape(img, shapes.row(0), cv::Scalar(255, 0, 0));
    aam::drawShape(img, shapes.row(1), cv::Scalar(0, 255, 0));
    cv::imshow("shapes before", img);
    cv::waitKey();
#endif
    
    
}


TEST_CASE("generalized-procustes")
{
    aam::RowVector2 mean;
    mean << -2.f, -2.f;
    aam::MatrixX cov = generate2DCovarianceMatrixFromStretchAndRotation(3, 3, 0.0);

    aam::MatrixX X = sampleMultivariateGaussian(mean, cov, 5);

    // Note, Transform assumes column vectors, so we need to transpose it when applied to a matrix.
    Eigen::Transform<aam::Scalar, 2, Eigen::Affine> sim1, sim2;
    sim1 = Eigen::Rotation2D<aam::Scalar>(3.1415f) * Eigen::Scaling(5.f) * Eigen::Translation<aam::Scalar, 2>(-10.f, -10.f);
    sim2 = Eigen::Scaling(2.f) * Eigen::Translation<aam::Scalar, 2>(5.f, 5.f);

    aam::MatrixX Y = (X.rowwise().homogeneous() * sim1.matrix().transpose()).rowwise().hnormalized();
    aam::MatrixX Z = (X.rowwise().homogeneous() * sim2.matrix().transpose()).rowwise().hnormalized();

    // Generalized procrustes expects a single row per shape.
    aam::MatrixX C(3, Y.rows() * 2);
    for (aam::MatrixX::Index i = 0; i < X.rows(); ++i) {
        C(0, i * 2 + 0) = X(i, 0);
        C(0, i * 2 + 1) = X(i, 1);
        C(1, i * 2 + 0) = Y(i, 0);
        C(1, i * 2 + 1) = Y(i, 1);
        C(2, i * 2 + 0) = Z(i, 0);
        C(2, i * 2 + 1) = Z(i, 1);
    }


    aam::generalizedProcrustes(C, 10);
    REQUIRE(C.row(0).isApprox(C.row(1)));
    REQUIRE(C.row(0).isApprox(C.row(2)));
}
