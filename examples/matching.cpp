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

#include <aam/aam.h>
#include <aam/matcher.h>

#include <opencv2/highgui/highgui.hpp>
#include <iostream>

/**
 
 Main entry point.
 
 */
int main(int argc, char **argv)
{
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " directory" << std::endl;
        return 0;
    }

    aam::TrainingSet trainingSet;
    aam::loadAsfTrainingSet(argv[1], trainingSet);
    aam::Trainer::createTriangulation(trainingSet);

    //aam::showTrainingSet(trainingSet);

    aam::ActiveAppearanceModel model;
    aam::Trainer trainer(trainingSet);
    trainer.train(model);

    //model.save("model.data");
    //model.load("model.data");

    aam::Matcher matcher(model);
    aam::Affine2 pose;
    aam::RowVectorX shapeParams;
    aam::RowVectorX appearanceParams;
    cv::Mat image = trainingSet.images[13].clone();
    matcher.match(image, pose, shapeParams, appearanceParams);

    /*
    cv::Mat imgShowAppearance = trainingSet.images[0].clone();
    model.renderAppearanceInstanceToImage(imgShowAppearance, pose, shapeParams, appearanceParams);

    cv::Mat imgShowShape = trainingSet.images[0].clone();
    model.renderShapeInstanceToImage(imgShowShape, pose, shapeParams);

    cv::imshow("Image", image);
    cv::imshow("MatchedAppearance", imgShowAppearance);
    cv::imshow("MatchedShape", imgShowShape);
    cv::waitKey(0);
    */

	return 0;
}




