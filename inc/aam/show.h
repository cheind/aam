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

namespace aam {

// for debugging: display a single training data (image + shape)
bool showTrainingData(const TrainingData& trainingData);

// for debugging: display the complete training set
bool showTrainingSet(const TrainingSet& trainingSet);

// TODO
//void training(std::vector<TrainingData> data, cv::Mat& eigenVecs, cv::Mat& eigenValues, cv::Mat& mean, double maxPercentVariation, ...);

}

#endif