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
#include <aam/bilinear.h>
#include <iostream>


TEST_CASE("bilinear")
{
    cv::Mat img(4, 4, CV_8UC4);
    cvSet2D(&(IplImage)img, 0, 0, cv::Scalar(255, 0, 0, 0));
    cvSet2D(&(IplImage)img, 0, 1, cv::Scalar(0, 255, 0, 0));
    cvSet2D(&(IplImage)img, 1, 0, cv::Scalar(0, 0, 255, 0));
    cvSet2D(&(IplImage)img, 1, 1, cv::Scalar(0, 0, 0, 255));

    // Pixel centers
    REQUIRE(aam::bilinear(img, 0.5, 0.5) == cv::Scalar(255, 0, 0, 0));
    REQUIRE(aam::bilinear(img, 0.5, 1.5) == cv::Scalar(0, 255, 0, 0));
    REQUIRE(aam::bilinear(img, 1.5, 0.5) == cv::Scalar(0, 0, 255, 0));
    REQUIRE(aam::bilinear(img, 1.5, 1.5) == cv::Scalar(0, 0, 0, 255));

    // Off-centers
    REQUIRE(aam::bilinear(img, 0.5, 1.0) == cv::Scalar(127.5, 127.5, 0, 0));
    REQUIRE(aam::bilinear(img, 1.0, 0.5) == cv::Scalar(127.5, 0, 127.5, 0));    
    REQUIRE(aam::bilinear(img, 1.0, 1.0) == cv::Scalar(63.75, 63.75, 63.75, 63.75));
    
    // Out-of-bounds
    REQUIRE(aam::bilinear(img, 0.5, 0) == cv::Scalar(127.5, 127.5, 0, 0));
    REQUIRE(aam::bilinear(img, 0, 0.5) == cv::Scalar(127.5, 0, 127.5, 0));


}