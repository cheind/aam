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


#include <aam/types.h>
#include <opencv/highgui.h>

/*
void aam::delaunayTriangulation(const cv::Mat& coords, cv::Mat& triangles) {
    cv::Subdiv2D subd;
    for (int i = 0; i < coords.rows; i++) {
        subd.insert(cv::Point2d(coords.at<double>(i, 0), coords.at<double>(i, 1)));
    }
    std::vector<cv::Vec6f> triangleList;
    subd.getTriangleList(triangleList);
        
    // TODO
    //cv::PCA pca();
}
*/

bool aam::showTrainingData(const TrainingData& trainingData) {
        
    cv::Mat dispImg;
    trainingData.img.copyTo(dispImg);
    for (int j = 0; j < trainingData.coords.rows; j++) {
        float x = (float)(trainingData.coords.at<double>(j, 0) * (double)dispImg.cols);
        float y = (float)(trainingData.coords.at<double>(j, 1) * (double)dispImg.rows);
        cv::circle(dispImg, cv::Point2f(x, y), 3, cv::Scalar(255), 2);

        //int c1 = pimpl->trainingSet[i].triangles.at<unsigned short>(j, 0);
        int c1 = trainingData.contours.at<unsigned short>(j, 1);
        int c2 = trainingData.contours.at<unsigned short>(j, 2);
        float x1 = (float)(trainingData.coords.at<double>(c1, 0) * (double)dispImg.cols);
        float y1 = (float)(trainingData.coords.at<double>(c1, 1) * (double)dispImg.rows);
        float x2 = (float)(trainingData.coords.at<double>(c2, 0) * (double)dispImg.cols);
        float y2 = (float)(trainingData.coords.at<double>(c2, 1) * (double)dispImg.rows);
        cv::line(dispImg, cv::Point2f(x, y), cv::Point2f(x1, y1), cv::Scalar(255), 1);
        cv::line(dispImg, cv::Point2f(x, y), cv::Point2f(x2, y2), cv::Scalar(255), 1);
    }
    cv::imshow("img", dispImg);
    cv::waitKey(0);

    return true;
}

bool aam::showTrainingSet(const TrainingSet& trainingSet) {
    for (size_t i = 0; i < trainingSet.size(); i++) {
        showTrainingData(trainingSet[i]);
    }
    return true;
}
