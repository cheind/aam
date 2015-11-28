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
#include <aam/trainer.h>
#include <aam/io.h>
#include <aam/show.h>
#include <aam/barycentrics.h>
#include <aam/rasterization.h>
#include <aam/map.h>

#include <opencv2/highgui/highgui.hpp>
#include <iostream>


void printMat(aam::MatrixX m) {
    for (int i = 0; i < m.rows(); i++) {
        for (int j = 0; j < m.cols(); j++) {
            std::cout << m(i, j) << " ";
        }
        std::cout << std::endl;
    }
}


void visualize(aam::ActiveAppearanceModel& model) {

    aam::MatrixX trafo = model.shapeTransformToTrainingData;
    //trafo(2, 0) -= 200;
    //trafo(2, 1) -= 200;

    aam::RowVectorX s = aam::transformShape(trafo, model.shapeMean);

    cv::Mat colors = aam::toOpenCVHeader<aam::Scalar>(model.appearanceMean.transpose());
    cv::Mat image = cv::Mat(640, 480, CV_8U);
    image.setTo(0);
    aam::writeShapeImage(s, model.triangleIndices, model.barycentricSamplePositions, colors, image);

    printMat(model.shapeModeWeights.rightCols(1));
    printMat(model.shapeModes.bottomRows(1));

    aam::Scalar w = 0.00005 * sqrt(model.shapeModeWeights.rightCols(1)(0, 0));

    
    aam::RowVectorX s2 = aam::transformShape(trafo, model.shapeMean + w * model.shapeModes.bottomRows(1));
    std::cout << "rasterizeShape..." << std::endl;
    aam::MatrixX barys = aam::rasterizeShape(s2, model.triangleIndices, 640, 480);

    cv::Mat s2Texture;
    cv::Mat image2 = cv::Mat(640, 480, CV_8U);
    image2.setTo(0);
    std::cout << "read shape image..." << std::endl;
    // read texture image from mean shape image
    aam::readShapeImage(s, model.triangleIndices, barys, image, s2Texture);
    std::cout << "write shape image..." << std::endl;
    aam::writeShapeImage(s2, model.triangleIndices, barys, s2Texture, image2);

    aam::drawShapeLandmarks(image, s, cv::Scalar(0));
    cv::imshow("img1", image);

    aam::drawShapeLandmarks(image2, s2, cv::Scalar(0));
    cv::imshow("img2", image2);
    cv::waitKey(0);
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

    aam::Trainer trainer(trainingSet);
    aam::ActiveAppearanceModel model;
    trainer.train(model);

    model.save("test.model");
    
    //aam::ActiveAppearanceModel model;
    //model.load("test.model");

    visualize(model);

    //aam::showTrainingSet(trainingSet);

    return 0;
}




