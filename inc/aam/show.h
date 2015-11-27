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

#ifndef AAM_SHOW_H
#define AAM_SHOW_H

#include <aam/fwd.h>
#include <aam/types.h>
#include <opencv2/core/core.hpp>

namespace aam {
    
    // for debugging draw a single shape on some image
    void drawShapeLandmarks(cv::Mat& canvas, Eigen::Ref<RowVectorX const> shape, const cv::Scalar &color);

    /** Draw shape contour */
    void drawShapeContour(cv::Mat& canvas, Eigen::Ref<RowVectorX const> shape, const cv::Mat& contourIds, const cv::Scalar &color);

    /** Draw shape triangles */
    void drawShapeTriangulation(cv::Mat& canvas, Eigen::Ref<RowVectorX const> shape, Eigen::Ref<RowVectorXi const> triangleIds, const cv::Scalar &color);
    
    // for debugging: display the complete training set
    void showTrainingSet(const TrainingSet& trainingSet);

}

#endif