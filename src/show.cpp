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


#include <aam/show.h>
#include <aam/delaunay.h>
#include <aam/map.h>
#include <aam/trainingset.h>
#include <opencv2/opencv.hpp>
#include <iostream>

namespace aam {
    void drawShapeLandmarks(cv::Mat& canvas, Eigen::Ref<RowVectorX const> shape, const cv::Scalar &color) {

        const auto &s = shape.cast<float>();
        
        for (int j = 0; j < s.cols() / 2; j++) {
            float x = s(0, j * 2 + 0);
            float y = s(0, j * 2 + 1);
            cv::circle(canvas, cv::Point2f(x, y), 2, color, 1, CV_AA);
        }
    }

    /** Draw shape contour */
    void drawShapeContour(cv::Mat& canvas, Eigen::Ref<RowVectorX const> shape, const cv::Mat& contourIds, const cv::Scalar &color)
    {
        const auto &s = shape.cast<float>();
        
        for (int j = 0; j < s.cols() / 2; j++) {
            float x = s(0, j * 2 + 0);
            float y = s(0, j * 2 + 1);

            int c1 = contourIds.at<int>(j, 1);
            int c2 = contourIds.at<int>(j, 2);
            float x1 = s(0, c1 * 2 + 0);
            float y1 = s(0, c1 * 2 + 1);
            float x2 = s(0, c2 * 2 + 0);
            float y2 = s(0, c2 * 2 + 1);
            cv::line(canvas, cv::Point2f(x, y), cv::Point2f(x1, y1), color, 1, CV_AA);
            cv::line(canvas, cv::Point2f(x, y), cv::Point2f(x2, y2), color, 1, CV_AA);
        }
    }

    /** Draw shape triangles */
    void drawShapeTriangulation(cv::Mat& canvas, Eigen::Ref<RowVectorX const> shape, Eigen::Ref<RowVectorXi const> triangleIds, const cv::Scalar &color)
    {
        const auto &s = shape.cast<float>();
        
        for (int j = 0; j < triangleIds.cols() / 3; j++) {

            int id1 = triangleIds(0, j * 3 + 0);
            int id2 = triangleIds(0, j * 3 + 1);
            int id3 = triangleIds(0, j * 3 + 2);

            float x1 = s(0, id1 * 2 + 0);
            float y1 = s(0, id1 * 2 + 1);

            float x2 = s(0, id2 * 2 + 0);
            float y2 = s(0, id2 * 2 + 1);

            float x3 = s(0, id3 * 2 + 0);
            float y3 = s(0, id3 * 2 + 1);

            cv::line(canvas, cv::Point2f(x1, y1), cv::Point2f(x2, y2), color, 1, CV_AA);
            cv::line(canvas, cv::Point2f(x2, y2), cv::Point2f(x3, y3), color, 1, CV_AA);
            cv::line(canvas, cv::Point2f(x3, y3), cv::Point2f(x1, y1), color, 1, CV_AA);
        }
    }

    void showTrainingSet(const aam::TrainingSet& trainingSet) {


        for (int i = 0; i < (int)trainingSet.images.size(); i++) {
            cv::Mat dispImg;
            trainingSet.images[i].copyTo(dispImg);
            drawShapeLandmarks(dispImg, trainingSet.shapes.row(i), cv::Scalar::all(255));
            if (trainingSet.triangles.size() > 0)
                drawShapeTriangulation(dispImg, trainingSet.shapes.row(i), trainingSet.triangles, cv::Scalar::all(255));
            cv::imshow("Training Image", dispImg);
            cv::waitKey(0);
        }
    }
}