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
#include <opencv/highgui.h>
#include <iostream>

namespace aam {
    void drawShapeLandmarks(cv::Mat& canvas, const cv::Mat& shape, const cv::Scalar &color) {

        for (int j = 0; j < shape.cols / 2; j++) {
            Scalar x = (shape.at<Scalar>(0, j * 2 + 0) * (Scalar)canvas.cols);
            Scalar y = (shape.at<Scalar>(0, j * 2 + 1) * (Scalar)canvas.rows);
            cv::circle(canvas, cv::Point2f(x, y), 2, color, 1, CV_AA);
        }
    }

    /** Draw shape contour */
    void drawShapeContour(cv::Mat& canvas, const cv::Mat& shape, const cv::Mat& contourIds, const cv::Scalar &color)
    {
        for (int j = 0; j < shape.cols / 2; j++) {
            Scalar x = (shape.at<Scalar>(0, j * 2 + 0) * (Scalar)canvas.cols);
            Scalar y = (shape.at<Scalar>(0, j * 2 + 1) * (Scalar)canvas.rows);

            int c1 = contourIds.at<unsigned short>(j, 1);
            int c2 = contourIds.at<unsigned short>(j, 2);
            Scalar x1 = (shape.at<Scalar>(0, c1 * 2 + 0) * (Scalar)canvas.cols);
            Scalar y1 = (shape.at<Scalar>(0, c1 * 2 + 1) * (Scalar)canvas.rows);
            Scalar x2 = (shape.at<Scalar>(0, c2 * 2 + 0) * (Scalar)canvas.cols);
            Scalar y2 = (shape.at<Scalar>(0, c2 * 2 + 1) * (Scalar)canvas.rows);
            cv::line(canvas, cv::Point2f(x, y), cv::Point2f(x1, y1), color, 1, CV_AA);
            cv::line(canvas, cv::Point2f(x, y), cv::Point2f(x2, y2), color, 1, CV_AA);
        }
    }

    /** Draw shape triangles */
    void drawShapeTriangulation(cv::Mat& canvas, const cv::Mat& shape, const cv::Mat& triangleIds, const cv::Scalar &color)
    {
        for (int j = 0; j < triangleIds.cols / 3; j++) {

            int id1 = triangleIds.at<int>(0, j * 3 + 0);
            int id2 = triangleIds.at<int>(0, j * 3 + 1);
            int id3 = triangleIds.at<int>(0, j * 3 + 2);

            Scalar x1 = (shape.at<Scalar>(0, id1 * 2 + 0) * (Scalar)canvas.cols);
            Scalar y1 = (shape.at<Scalar>(0, id1 * 2 + 1) * (Scalar)canvas.rows);

            Scalar x2 = (shape.at<Scalar>(0, id2 * 2 + 0) * (Scalar)canvas.cols);
            Scalar y2 = (shape.at<Scalar>(0, id2 * 2 + 1) * (Scalar)canvas.rows);

            Scalar x3 = (shape.at<Scalar>(0, id3 * 2 + 0) * (Scalar)canvas.cols);
            Scalar y3 = (shape.at<Scalar>(0, id3 * 2 + 1) * (Scalar)canvas.rows);

            cv::line(canvas, cv::Point2f(x1, y1), cv::Point2f(x2, y2), color, 1, CV_AA);
            cv::line(canvas, cv::Point2f(x2, y2), cv::Point2f(x3, y3), color, 1, CV_AA);
            cv::line(canvas, cv::Point2f(x3, y3), cv::Point2f(x1, y1), color, 1, CV_AA);
        }
    }

    void showTrainingSet(const aam::TrainingSet& trainingSet) {

        aam::RowVectorXi triIds = aam::findDelaunayTriangulation(toEigenHeader<float>(trainingSet.shapes.rowRange(0, 1)));
        cv::Mat triIdsCV = toOpenCVHeader<int>(triIds);

        for (int i = 0; i < (int)trainingSet.images.size(); i++) {
            cv::Mat dispImg;
            trainingSet.images[i].copyTo(dispImg);
            drawShapeLandmarks(dispImg, trainingSet.shapes.rowRange(i, i + 1), cv::Scalar::all(255));
            //if (!trainingSet.contour.empty())
            //    drawShapeContour(dispImg, trainingSet.shapes.rowRange(i, i + 1), trainingSet.contour, cv::Scalar::all(255));
            drawShapeTriangulation(dispImg, trainingSet.shapes.rowRange(i, i + 1), triIdsCV, cv::Scalar::all(255));
            cv::imshow("img", dispImg);
            cv::waitKey(0);
        }
    }
}