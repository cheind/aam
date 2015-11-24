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
#include <opencv/highgui.h>

namespace aam {
    void drawShape(cv::Mat& canvas, const cv::Mat& shape, const cv::Mat& contour) {

        for (int j = 0; j < shape.cols / 2; j++) {
            Scalar x = (shape.at<Scalar>(0, j * 2 + 0) * (Scalar)canvas.cols);
            Scalar y = (shape.at<Scalar>(0, j * 2 + 1) * (Scalar)canvas.rows);
            cv::circle(canvas, cv::Point2f(x, y), 2, cv::Scalar(255), 1, CV_AA);

            if (!contour.empty()) {
                int c1 = contour.at<unsigned short>(j, 1);
                int c2 = contour.at<unsigned short>(j, 2);
                Scalar x1 = (shape.at<Scalar>(0, c1 * 2 + 0) * (Scalar)canvas.cols);
                Scalar y1 = (shape.at<Scalar>(0, c1 * 2 + 1) * (Scalar)canvas.rows);
                Scalar x2 = (shape.at<Scalar>(0, c2 * 2 + 0) * (Scalar)canvas.cols);
                Scalar y2 = (shape.at<Scalar>(0, c2 * 2 + 1) * (Scalar)canvas.rows);
                cv::line(canvas, cv::Point2f(x, y), cv::Point2f(x1, y1), cv::Scalar(255), 1, CV_AA);
                cv::line(canvas, cv::Point2f(x, y), cv::Point2f(x2, y2), cv::Scalar(255), 1, CV_AA);
            }
        }
    }

    void showTrainingSet(const aam::TrainingSet& trainingSet) {
        for (int i = 0; i < (int)trainingSet.images.size(); i++) {
            cv::Mat dispImg;
            trainingSet.images[i].copyTo(dispImg);
            drawShape(dispImg, trainingSet.shapes.rowRange(i, i + 1), trainingSet.contour);

            cv::imshow("img", dispImg);
            cv::waitKey(0);
        }
    }
}