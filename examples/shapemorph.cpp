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

/**
 
 Main entry point.
 
 */
int main(int argc, char **argv)
{
    aam::TrainingSet trainingSet;
    aam::loadAsfTrainingSet("c:/GIT/AAM_data/data", trainingSet);

    // calculate PCA on shape data (without procrustes analysis for now...)
    cv::Mat mean;
    cv::PCA pca(trainingSet.shapes, mean, 0, 0.1);
    cv::Mat vecs = pca.eigenvectors;
    cv::Mat vals = pca.eigenvalues;
    mean = pca.mean;

    // display bilinear interpolation between first two shapes
    cv::Mat dispImg = cv::Mat(480, 640, CV_8U);
    int key = 0;
    int counter = 0;
    while (key != 27) {
        dispImg = cv::Scalar(0);
        double w0 = sin((double)counter / 20.0);
        double w1 = sin((double)counter / 24.0);
        //cv::Mat s = trainingSet.shapes.rowRange(0, 1) * w0 + trainingSet.shapes.rowRange(5, 6) * w1;

        cv::Mat s = mean + w0 * vecs.rowRange(0, 1) + w1 * vecs.rowRange(1, 2);

        aam::drawShape(dispImg, s, trainingSet.contour);
        cv::imshow("img", dispImg);
        std::cout << "w = " << w0 << std::endl;
        key = cv::waitKey(10);
        counter++;
    }

    /* TODO
    aam::procrustes(M_in, M_out);  // M...PxQ elements with P = number of shapes, Q = number of coordinates per shape
                                   // output: same matrix as input but shaped aligned (i.e. modulo translation, scaling and rotation)

    aam::pca(M_out, mean_out, basis_out, weights_out);

    //aam::barySample(imgs, tex_Mat_out)
    //aam::pca(tex_Mat_out)

    for (x = -weights[0] : weights[0]) {
        s = s0 + x * basis_out
        show(s);
    }
    */

	return 0;
}




