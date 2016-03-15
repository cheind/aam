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

#include <time.h>

#include <aam/aam.h>
#include <aam/matcher.h>

#include <opencv2/highgui/highgui.hpp>
#include <iostream>

void printModelInfo(const aam::ActiveAppearanceModel& model) {
    std::cout << "Active Appearance Model:" << std::endl;
    std::cout << "  number of shape modes: " << model.shapeModes.rows() << std::endl;
    std::cout << "  number of appearance modes: " << model.appearanceModes.rows() << std::endl;
}

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

#define BUILD_MODEL  // comment this line and re-compile to load existing model (faster start-up in debug-mode)
#ifdef BUILD_MODEL  // build the model from the training data
    aam::Trainer trainer(trainingSet);
    trainer.train(model);
    model.save("model.data");

    printModelInfo(model);
    model.setNumShapeModes(3);
    model.setNumAppearanceModes(15);
    printModelInfo(model);

#else  // load saved model
    model.load("model.data");
#endif

    aam::Matcher2 matcher(model);
    aam::Affine2 pose;
    aam::RowVectorX shapeParams = aam::RowVectorX::Zero(1, model.shapeModeWeights.cols());
    aam::RowVectorX appearanceParams = aam::RowVectorX::Zero(1, model.appearanceModeWeights.cols());

    int nbTrainingExamples = (int)trainingSet.images.size();
    cv::Mat image = trainingSet.images[6 * 5].clone();  // use the 18-th face from the training set
    //cv::Mat image = cv::imread("c:/GIT/AAM_data/szamba.png", 0);
    //cv::Mat image = cv::imread("c:/GIT/AAM_data/cheind.png", 0);
    //cv::Mat image = cv::imread("c:/data/dev/aam_data/test_001.jpg", 0);

    std::cout << "init matcher..." << std::endl;

    // initialize the AAM matcher
    srand((unsigned int)time(0));
    aam::Scalar x = 320;
    aam::Scalar y = 240;
    aam::Scalar scaling = 1.0;
    matcher.init(image, x, y, scaling, shapeParams, appearanceParams);

    std::cout << "press 'a' to match without further keypress" << std::endl;
    std::cout << "press other key to match step by step" << std::endl;
    std::cout << "press Escape to quit" << std::endl;

    //TODO: do something like "while(error > epsilon)"
    int key = 0;
    int delay = 0;
    while (key != 27) {

        aam::Affine2 currentWarp = matcher.getCurrentGlobalTransform();

        // visualize the current model instance and wait for key press
        cv::Mat imgShowAppearance = image.clone();
        model.renderAppearanceInstanceToImage(imgShowAppearance, currentWarp, matcher.getCurrentShapeParams(), matcher.getCurrentAppearanceParams());
        cv::Mat imgShowShape = image.clone();
        model.renderShapeInstanceToImage(imgShowShape, currentWarp, matcher.getCurrentShapeParams());

        cv::imshow("Image", image);
        cv::imshow("MatchedAppearance", imgShowAppearance);
        cv::imshow("MatchedShape", imgShowShape);
        key = cv::waitKey(delay);
        if (key == 'a') {
            delay = 20;
        }

        // make a single matching step
        matcher.step();
    }

	return 0;
}




