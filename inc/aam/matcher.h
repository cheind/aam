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

#ifndef AAM_MATCHER_H
#define AAM_MATCHER_H

#include <aam/fwd.h>
#include <aam/types.h>
#include <aam/model.h>

namespace aam {
   
    /** class for matching an AAM using the inverse compositional approach */
    class Matcher {

    private:

        /** The model that is matched to images */
        ActiveAppearanceModel model;

        /** the input image to which the model is matched */
        cv::Mat image;

        /** pre-computed gradient images, matrices are 1x2 */
        std::vector<MatrixX> grad;

        /** pre-computed jacobians of warp, matrices are 2x4 */
        std::vector<MatrixX> jacobians;

        /** pre-computed steepest descent images, matrices are 1x4 */
        std::vector<MatrixX> steepestDecentImgs;

        /** pre-computed inverse hessian, matrix is 4x4 */
        MatrixX invHessian;

        /** pre-computed cartesian coordinates of sample positions (relative to mean shape) */
        std::vector<RowVector2> coords;

        /** current warp */
        Affine2 currentWarp;

    public:

        /** Constructor */
        Matcher(const aam::ActiveAppearanceModel& model);

        /** Initialize the matching (i.e. pre-compute various entities) */
        void init(const cv::Mat& img, aam::Affine2& pose, aam::RowVectorX& shapeParams, aam::RowVectorX& textureParams);

        /** match the active appearance model to the given image */
        void step();

        /** returns the current warp */
        Affine2 getCurrentGlobalTransform();
    };

}

#endif