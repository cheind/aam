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

#include <aam/io.h>
#include <aam/types.h>
#include <aam/trainingset.h>
#include <aam/views.h>

#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>

bool parseAsfFile(const std::string& fileName, aam::RowVectorX &coords, cv::Mat& contours) {

    int landmarkCount = 0;

    std::ifstream file(fileName);

    if (!file.is_open()) {
        return false;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.size() > 0) {
            if (line[0] != '#') {
                if (line.find(".jpg") != std::string::npos) {
                    // ignored: file name of jpg image
                }
                else if (line.size() < 10) {
                    int nbPoints = atol(line.c_str());
                    coords.resize(1, nbPoints * 2);
                    contours = cv::Mat(nbPoints, 3, CV_32SC1);
                }
                else {
                    std::stringstream stream(line);
                    std::string path;
                    std::string type;
                    std::string xStr;
                    std::string yStr;
                    std::string pointIdStr;
                    std::string conn1Str;
                    std::string conn2Str;
                    stream >> path;
                    stream >> type;
                    stream >> xStr;
                    stream >> yStr;
                    stream >> pointIdStr;
                    stream >> conn1Str;
                    stream >> conn2Str;

                    aam::Scalar x = (aam::Scalar)atof(xStr.c_str());
                    aam::Scalar y = (aam::Scalar)atof(yStr.c_str());
                    
                    int id = atoi(pointIdStr.c_str());
                    int c1 = atoi(conn1Str.c_str());
                    int c2 = atoi(conn2Str.c_str());

                    coords(0, landmarkCount * 2 + 0) = x;
                    coords(0, landmarkCount * 2 + 1) = y;

                    contours.at<int>(landmarkCount, 0) = id;
                    contours.at<int>(landmarkCount, 1) = c1;
                    contours.at<int>(landmarkCount, 2) = c2;

                    landmarkCount++;
                }
            }
        }
    }
    return true;
}

bool loadTrainingExample(const std::string& directory, int majorIndex, int minorIndex, const std::string& sex, cv::Mat& image, aam::RowVectorX& shape, cv::Mat& contour) {
    char *name = new char[directory.size() + 100];
#ifdef _WIN32
    sprintf_s(name, directory.size() + 100, "%s/%02d-%d%s", directory.c_str(), majorIndex, minorIndex, sex.c_str());
#else
    sprintf(name, "%s/%02d-%d%s", directory.c_str(), majorIndex, minorIndex, sex.c_str());
#endif
    std::string fileNameImg = std::string(name) + ".jpg";
    std::string fileNamePts = std::string(name) + ".asf";
    
    image = cv::imread(fileNameImg, 0);
    std::cout << "loading " << fileNameImg << ", " << fileNamePts << std::endl;
    
    delete[] name;

    return parseAsfFile(fileNamePts, shape, contour);
}

bool aam::loadAsfTrainingSet(const std::string& directory, aam::TrainingSet& trainingSet, int firstNExamplesToLoad) {

    trainingSet.images.clear();

    std::vector<aam::RowVectorX> shapeVecs;

    cv::Mat contour;
    
    bool ok = true;
    int i = 1;
    int counter = 0;
    do {
        bool subIdOK = true;
        int j = 1;
        do {
            cv::Mat image;
            aam::RowVectorX shape;
            if (!loadTrainingExample(directory, i, j, "m", image, shape, contour)) {
                loadTrainingExample(directory, i, j, "f", image, shape, contour);
            }

            if (shape.cols() > 0) {
                shapeVecs.push_back(shape);
                trainingSet.images.push_back(image);
                j++;
                counter++;
            }

            else {
                subIdOK = false;
            }

            if ((firstNExamplesToLoad > 0) && (counter >= firstNExamplesToLoad)) {
                ok = false;
            }
        } while (ok && subIdOK);
        if (j == 1) {
            ok = false;
        }
        i++;
    } while (ok);

    // assemble the complete shape matrix from all training shapes that are given as row vectors
    trainingSet.shapes.resize(shapeVecs.size(), shapeVecs[0].cols());
        
    for (size_t i = 0; i < shapeVecs.size(); ++i) {
        aam::toSeparatedView<Scalar>(shapeVecs[i]).col(0) *= (aam::Scalar)trainingSet.images[i].cols;
        aam::toSeparatedView<Scalar>(shapeVecs[i]).col(1) *= (aam::Scalar)trainingSet.images[i].rows;
        trainingSet.shapes.row(i) = shapeVecs[i];
    }
    trainingSet.contour = contour;

    return ok;
}