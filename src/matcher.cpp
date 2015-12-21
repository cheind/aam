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


#include <aam/matcher.h>
#include <aam/fwd.h>

#include <imagealign/imagealign.h>

#include <opencv2/highgui/highgui.hpp>

namespace ia = imagealign;

namespace aam {

    Matcher::Matcher(const aam::ActiveAppearanceModel& model) {
        this->model = model;
    }

    void setInvalidPixelsToZero(cv::Mat& image, const cv::Mat& mask) {
        for (int i = 1; i < image.rows-1; i++) {
            for (int j = 1; j < image.cols-1; j++) {
                if (mask.at<unsigned char>(i, j) == 0) {
                    image.at<float>(i, j) = 0;
                    image.at<float>(i-1, j) = 0;
                    image.at<float>(i, j-1) = 0;
                    image.at<float>(i+1, j) = 0;
                    image.at<float>(i, j+1) = 0;

                    image.at<float>(i-1, j-1) = 0;
                    image.at<float>(i+1, j-1) = 0;
                    image.at<float>(i-1, j+1) = 0;
                    image.at<float>(i+1, j+1) = 0;
                }
            }
        }
    }

    void calcGradientOfMeanAppearance(const cv::Mat& image, ActiveAppearanceModel& model, cv::Mat& gradX, cv::Mat& gradY) {
        
        cv::Mat meanTextureImage = cv::Mat(image.rows, image.cols, CV_8U);
        meanTextureImage = cv::Scalar(0);
        model.renderAppearanceInstanceToImage(meanTextureImage, model.shapeTransformToTrainingData, MatrixX::Zero(1, model.shapeModeWeights.cols()), MatrixX::Zero(1, model.appearanceModeWeights.cols()), false);
        cv::Sobel(meanTextureImage, gradX, CV_32F, 1, 0, 3);
        cv::Sobel(meanTextureImage, gradY, CV_32F, 0, 1, 3);
        setInvalidPixelsToZero(gradX, meanTextureImage);
        setInvalidPixelsToZero(gradY, meanTextureImage);
        cv::imshow("gradX", gradX * (1.0/512) + 0.5);
        cv::imshow("gradY", gradY * (1.0/512) + 0.5);
        //cv::waitKey(0);
    }

    //void evaluateJacobian(ActiveAppearanceModel& model, MatrixX jacobian) {  
    //}

    void Matcher::match(const cv::Mat& image, aam::Affine2& pose, aam::RowVectorX& shapeParams, aam::RowVectorX& textureParams) {

        // initialize similarity transform (i.e. pose), shape parameters, and texture parameters
        pose = model.shapeTransformToTrainingData;
        aam::Scalar scaleFactor = (aam::Scalar)1.3;  // 1.3
        pose(0, 0) *= (aam::Scalar)scaleFactor;
        pose(1, 1) *= (aam::Scalar)scaleFactor;
        pose(2, 0) += (aam::Scalar)55; // 55
        pose(2, 1) += (aam::Scalar)-10; // -10
        shapeParams = MatrixX::Zero(1, model.shapeModeWeights.cols());
        textureParams = MatrixX::Zero(1, model.appearanceModeWeights.cols());

        // calculate the gradient of the template (i.e. mean appearance)
        cv::Mat gradX;
        cv::Mat gradY;
        calcGradientOfMeanAppearance(image, model, gradX, gradY);

        // evaluate the Jacobian at (x; 0)
        //evaluateJacobian(...);
        // TODO

        // compute steepest descent images grad(A_0) dW/dp
        // TODO

        // compute the Hessian matrix (eq. 23)
        // TODO

        // Loop
        //    warp I with W(x; p) to compute I(W; p)
        //    Compute the error image I(W(x; p)) - A_0(x)
        //    ...



        for (int i = 0; i < 100; i++) {

            shapeParams(0, shapeParams.cols()-2) += (aam::Scalar)0.3 * model.shapeModeWeights(0, shapeParams.cols()-2);

            // Visualize result...
            cv::Mat imgShowAppearance = image.clone();
            //model.renderAppearanceInstanceToImage(imgShowAppearance, pose, shapeParams, textureParams);
            cv::Mat imgShowShape = image.clone();
            model.renderShapeInstanceToImage(imgShowShape, pose, shapeParams);
            cv::imshow("Image", image);
            //cv::imshow("MatchedAppearance", imgShowAppearance);
            cv::imshow("MatchedShape", imgShowShape);
            cv::waitKey(0);
        }
    }

}
