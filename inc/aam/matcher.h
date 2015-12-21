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
    public:

        /** The model that is matched to images */
        ActiveAppearanceModel model;

        /** Constructor */
        Matcher(const aam::ActiveAppearanceModel& model);

        /** match the active appearance model to the given image */
        void match(const cv::Mat& image, aam::Matrix2& initialPose, aam::MatrixX& shapeParams, aam::MatrixX& textureParams);
    };

}

#endif