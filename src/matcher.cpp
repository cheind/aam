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

#include <opencv2/highgui/highgui.hpp>

namespace aam {

    Matcher::Matcher(const aam::ActiveAppearanceModel& model) {
        this->model = model;
    }

    void Matcher::match(const cv::Mat& image, aam::Affine2& pose, aam::RowVectorX& shapeParams, aam::RowVectorX& textureParams) {

        pose = model.shapeTransformToTrainingData;
        aam::Scalar scaleFactor = (aam::Scalar)1.3;
        pose(0, 0) *= (aam::Scalar)scaleFactor;
        pose(1, 1) *= (aam::Scalar)scaleFactor;
        pose(2, 0) += (aam::Scalar)55;
        pose(2, 1) += (aam::Scalar)-10;
        shapeParams = MatrixX::Zero(1, model.shapeModeWeights.cols());
        textureParams = MatrixX::Zero(1, model.appearanceModeWeights.cols());

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
