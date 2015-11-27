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

#ifndef AAM_TRAININGSET_H
#define AAM_TRAININGSET_H

#include <aam/types.h>

namespace aam {
    
    /** The complete training data that is needed to build/train an Active Appearance Model */
    class TrainingSet {
    public:
        std::vector<cv::Mat> images;       // training images
        cv::Mat shapes;    // NxM matrix with N (nb. rows) = number of training examples, M (nb. cols) = number of coordinates per training shape
        cv::Mat contour;  // optional: contours defined on the object (this data is just for visualization, not needed for actual AAM)
        aam::RowVectorXi triangles; // the triangles that span the shapes
    };
    
}

#endif