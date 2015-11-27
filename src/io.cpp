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

#include <iostream>
#include <fstream>
#include <opencv/highgui.h>

bool parseAsfFile(const std::string& fileName, cv::Mat& coords, cv::Mat& contours) {

    int landmarkCount = 0;

    std::ifstream file(fileName);

    std::string line;
    while (std::getline(file, line)) {
        if (line.size() > 0) {
            if (line[0] != '#') {
                if (line.find(".jpg") != std::string::npos) {
                    // ignored: file name of jpg image
                }
                else if (line.size() < 10) {
                    int nbPoints = atol(line.c_str());
                    //std::cout << "number of landmark points in file: " << nbPoints << std::endl;
                    coords = cv::Mat(1, nbPoints * 2, CV_MAKETYPE(cv::DataType<aam::Scalar>::depth, 1));
                    contours = cv::Mat(nbPoints, 3, CV_16U);
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

                    coords.at<aam::Scalar>(0, landmarkCount * 2 + 0) = x;
                    coords.at<aam::Scalar>(0, landmarkCount * 2 + 1) = y;

                    contours.at<unsigned short>(landmarkCount, 0) = id;
                    contours.at<unsigned short>(landmarkCount, 1) = c1;
                    contours.at<unsigned short>(landmarkCount, 2) = c2;

                    landmarkCount++;
                }
            }
        }
    }
    return true;
}

bool aam::loadAsfTrainingSet(const std::string& directory, aam::TrainingSet& trainingSet) {

    trainingSet.images.clear();

    std::vector<cv::Mat> shapeVecs;

    cv::Mat contour;
    
    bool ok = true;
    int i = 1;
    do {
        bool subIdOK = true;
        int j = 1;
        do {
            char *name = new char[directory.size() + 100];
#ifdef _WIN32
            sprintf_s(name, directory.size() + 100, "%s/%02d-%dm", directory.c_str(), i, j);
#else
            sprintf(name, "%s/%02d-%dm", directory.c_str(), i, j);
#endif
            std::string fileNameImg = std::string(name) + ".jpg";
            std::string fileNamePts = std::string(name) + ".asf";
            cv::Mat img = cv::imread(fileNameImg, 0);
            std::cout << "loading " << fileNameImg << ", " << fileNamePts << std::endl;
            cv::Mat coords;
            parseAsfFile(fileNamePts, coords, contour);
            if (coords.rows > 0) {
                shapeVecs.push_back(coords);
                trainingSet.images.push_back(img);
                j++;
            }
            else {
                subIdOK = false;
            }
            delete[] name;
        } while (subIdOK);
        if (j == 1) {
            ok = false;
        }
        i++;
    } while (ok);

    // assemble the complete shape matrix from all training shapes that are given as row vectors
    trainingSet.shapes = cv::Mat((int)shapeVecs.size(), shapeVecs[0].cols, CV_MAKETYPE(cv::DataType<aam::Scalar>::depth, 1));
    for (size_t i = 0; i < shapeVecs.size(); i++) {
        shapeVecs[i].copyTo(trainingSet.shapes.rowRange(i, i + 1));
    }

    trainingSet.contour = contour;

    return true;
}