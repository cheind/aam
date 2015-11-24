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

#define CATCH_CONFIG_MAIN  
#include "catch.hpp"

#include <aam/map.h>
#include <iostream>
#include <opencv2/opencv.hpp>

template<class EigenMatrix>
bool compareForMatricesForEquality(const cv::Mat m, const EigenMatrix &b)
{
    REQUIRE(m.rows == b.rows());
    REQUIRE(m.cols == b.cols());

    for (int y = 0; y < m.rows; ++y) {
        for (int x = 0; x < m.cols; ++x) {
            if (m.at<EigenMatrix::Scalar>(y, x) != Approx(b(y, x)))
                return false;
        }
    }

    return true;
}

TEST_CASE("map-header-only")
{
    aam::MatrixX mEigen = aam::MatrixX::Random(8, 6);

    // Map to OpenCV
    cv::Mat_<float> mOpenCVMapped = aam::toOpenCVHeader(mEigen);
    REQUIRE(compareForMatricesForEquality(mOpenCVMapped, mEigen));

    // Create non-contingous view of mat
    cv::Mat roi = mOpenCVMapped(cv::Rect(1, 1, 2, 2));

    // Map non-contingous back to eigen
    aam::MapMatrixX roiMapped = aam::toEigenHeader(roi);
    REQUIRE(compareForMatricesForEquality(roi, roiMapped));

    // Try with image data, 3 channels
    cv::Mat img(100, 100, CV_8UC3);
    
    // Some random lines
    cv::RNG rng;
    for (int i = 0; i < 100; i++)
    {
        cv::Point pt1, pt2;
        pt1.x = rng.uniform(0, img.cols);
        pt1.y = rng.uniform(0, img.rows);
        pt2.x = rng.uniform(0, img.cols);
        pt2.y = rng.uniform(0, img.rows);

        cv::line(img, pt1, pt2, cv::Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255)), rng.uniform(1, 10), 8);
    }

    auto eigenMapped = aam::toEigenHeader(cv::Mat_<uchar>(img));
    REQUIRE(eigenMapped.rows() == 100);
    REQUIRE(eigenMapped.cols() == 300); // Single channel support in Eigen
    REQUIRE(compareForMatricesForEquality(img.reshape(1), eigenMapped));
   
}