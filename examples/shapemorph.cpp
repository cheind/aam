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
#include <aam/io.h>

#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <iomanip>
#include <math.h>

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

    aam::Trainer trainer(trainingSet);
    aam::ActiveAppearanceModel model;
    trainer.train(model);

    //model.save("test.model");
    //model.load("test.model");

    cv::Mat img(480, 640, CV_8U);

    int key = 0;
    int counter = 0;
    do {
        img = cv::Scalar(0);

        aam::RowVectorX shapeParams = model.shapeModeWeights * 0;

        double deg = counter * 2;
        int mode = ((int)(deg / 360)) % 5 + 1;
        aam::Scalar eigenValue = model.shapeModeWeights(model.shapeModeWeights.cols() - mode);
        shapeParams(shapeParams.cols() - mode) = (aam::Scalar)(3 * sqrt(eigenValue) * std::sin(deg / 180 * 3.14156));

        model.renderShapeInstanceToImage(img, aam::MatrixX(0, 0), shapeParams);

        char str[100];
        sprintf(str, "mode %d", mode);
        cv::putText(img, str, cv::Point2i(100, 300), 1, 1, cv::Scalar(255));

        cv::imshow("AAM instance", img);
        key = cv::waitKey(10);

        counter++;
    } while(key != 27);
    
	return 0;
}




