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


#include <aam/matcher2.h>
#include <aam/fwd.h>
#include <aam/transform.h>

#include <imagealign/imagealign.h>

#include <Eigen/LU>
#include <Eigen/Dense>

#include <opencv2/highgui/highgui.hpp>
#include <time.h>
#include <stdlib.h>

namespace ia = imagealign;

namespace aam {

    Matcher2::Matcher2(const aam::ActiveAppearanceModel& model) {
        this->model = model;
    }

    Affine2 Matcher2::getCurrentGlobalTransform() {
        return currentWarp;
    }

    void setInvalidPixelsToZero(cv::Mat& image, const cv::Mat& mask) {
        for (int i = 1; i < image.rows-1; i++) {
            for (int j = 1; j < image.cols-1; j++) {
                if (mask.at<unsigned char>(i, j) == 0) {
                    image.at<float>(i, j) = 0;
                    image.at<float>(i-1, j) = 0;
                    image.at<float>(i, j-1) = 0;
                    image.at<float>(i+1, j) = 0;
                    image.at<float>(i, j+1) = 0;

                    image.at<float>(i-1, j-1) = 0;
                    image.at<float>(i+1, j-1) = 0;
                    image.at<float>(i-1, j+1) = 0;
                    image.at<float>(i+1, j+1) = 0;
                }
            }
        }
    }

    void calcGradientOfMeanAppearance(const cv::Mat& image, ActiveAppearanceModel& model, std::vector<aam::MatrixX>& grad) {
        
        cv::Mat meanTextureImage = cv::Mat(image.rows, image.cols, CV_8U);
        meanTextureImage = cv::Scalar(0);
        model.renderAppearanceInstanceToImage(meanTextureImage, model.shapeTransformToTrainingData, MatrixX::Zero(1, model.shapeModeWeights.cols()), MatrixX::Zero(1, model.appearanceModeWeights.cols()), false);
        cv::Mat gradX;
        cv::Mat gradY;
        cv::Sobel(meanTextureImage, gradX, CV_32F, 1, 0, 3);
        cv::Sobel(meanTextureImage, gradY, CV_32F, 0, 1, 3);
        
        setInvalidPixelsToZero(gradX, meanTextureImage);
        setInvalidPixelsToZero(gradY, meanTextureImage);
        cv::imshow("gradX", gradX * (1.0/512) + 0.5);
        cv::imshow("gradY", gradY * (1.0/512) + 0.5);

        std::vector<aam::RowVector2> cartesianCoords;
        model.getCartesianPixelCoordinates(model.shapeTransformToTrainingData, MatrixX::Zero(1, model.shapeModeWeights.cols()), cartesianCoords);

        grad.clear();
        for (size_t i = 0; i < cartesianCoords.size(); i++) {
            aam::MatrixX g(1, 2);
            int x = (int)cartesianCoords[i](0, 0);
            int y = (int)cartesianCoords[i](0, 1);
            g(0, 0) = gradX.at<float>(y, x);
            g(0, 1) = gradY.at<float>(y, x);
            grad.push_back(g);
        }
        
        //cv::waitKey(0);
    }

    void evaluateJacobiansGlobalTransform(ActiveAppearanceModel& model, std::vector<MatrixX>& jacobians) {  

        jacobians.clear();

        MatrixX s = model.shapeMean;

        for (int i = 0; i < model.barycentricSamplePositions.rows(); i++) {

            // get triangle, vertices and shape
            int triangleID = (int)model.barycentricSamplePositions(i, 0);
            aam::Scalar alpha = model.barycentricSamplePositions(i, 1);
            aam::Scalar beta = model.barycentricSamplePositions(i, 2);
            aam::Scalar gamma = 1 - (alpha + beta);
            int pt1idx = model.triangleIndices(0, triangleID * 3 + 0);
            int pt2idx = model.triangleIndices(0, triangleID * 3 + 1);
            int pt3idx = model.triangleIndices(0, triangleID * 3 + 2);
            
            // calculate x and y of the current pixel
            aam::Scalar x = s(0, pt1idx * 2 + 0) * alpha + s(0, pt2idx * 2 + 0) * beta + s(0, pt3idx * 2 + 0) * gamma;
            aam::Scalar y = s(0, pt1idx * 2 + 1) * alpha + s(0, pt2idx * 2 + 1) * beta + s(0, pt3idx * 2 + 1) * gamma;
            
            // calculate the Jacobian matrix
            AamMatrixTraits<Scalar, 2, 4>::MatrixType jacobian;
            jacobian(0, 0) = x;
            jacobian(0, 1) = y;
            jacobian(0, 2) = 1;
            jacobian(0, 3) = 0;
            jacobian(1, 0) = y;
            jacobian(1, 1) = x;
            jacobian(1, 2) = 0;
            jacobian(1, 3) = 1;

            // add the Jacobian for this pixel to the list
            jacobians.push_back(jacobian);
        }
    }

    void evaluateWarpJacobians(ActiveAppearanceModel& model, std::vector<MatrixX>& jacobians) {
        // TODO: implement equations 30 and 31
        // dW/dx_i * dx_i/dp = 
        //     image of influence of individual mesh vertices to relevant image region * 
        //     change of vertex per parameter (i.e. shape vectors re-arranged)
        // jacobians are 
    }

    void elementWiseMult(const std::vector<MatrixX>& a, const std::vector<MatrixX>& b, std::vector<MatrixX>& product) 
    {
        // TODO: make sure that a.size() == b.size(), otherwise: exception

        product.resize(a.size());
        for (size_t i = 0; i < a.size(); i++) {
            product[i] = a[i] * b[i];
        }
    }

    void calcInvHessian(const std::vector<MatrixX>& sd, MatrixX& invHessian) {
        invHessian = MatrixX::Zero(4, 4);

        for (size_t i = 0; i < sd.size(); i++) {
            invHessian += sd[i].adjoint() * sd[i];
        }

        invHessian = invHessian.inverse();
    }

    // convert parameter representation to affine transformation
    Affine2 paramsToWarp(const Eigen::Ref<MatrixX> params) {
        Affine2 retVal;
        
        // note: b and -b are swapped as we are doing multiplication from left side (i.e. row vectors)
        retVal(0, 0) = 1 + params(0, 0);   // 1 + a
        retVal(0, 1) = -params(1, 0);       // -b
        retVal(1, 0) = params(1, 0);      // b
        retVal(1, 1) = 1 + params(0, 0);   // 1 + a
        retVal(2, 0) = params(2, 0);       // t_x
        retVal(2, 1) = params(3, 0);       // t_y
        
        return retVal;
    }

    void Matcher2::init(const cv::Mat& img, aam::Affine2& pose, aam::RowVectorX& shapeParams, aam::RowVectorX& textureParams) {

        image = img.clone();

        // calculate the gradient of the template (i.e. mean appearance image)
        // gradients are 1x2
        calcGradientOfMeanAppearance(image, model, grad);

        // evaluate the global shape transform Jacobians at (x; 0)
        // jacobians are 2x4 for global shape transform
        evaluateJacobiansGlobalTransform(model, globalTrafoJacobians);

        // evaluate the warp Jacobians at (x; 0)
        // TODO
        evaluateWarpJacobians(model, warpJacobians);

        // compute steepest descent images grad(A_0) dW/dp
        // steepest descent images are 1x4
        elementWiseMult(grad, globalTrafoJacobians, steepestDecentImgs);

        // compute the inverse Hessian matrix (eq. 23)
        // inverse hessian is 4x4
        calcInvHessian(steepestDecentImgs, invHessian);

        // calculate cartesian sample positions of mean shape
        model.getCartesianPixelCoordinates(Affine2::Identity(), shapeParams, coords);

        // initialize the warp with the transform to training data
        currentWarp = model.shapeTransformToTrainingData;
        srand((unsigned int)time(0));
        currentWarp(2, 0) += 30 + (rand() % 40 - 20);  // for debugging only: shift in x-direction, TODO: remove!
        currentWarp(2, 1) += 10 + (rand() % 40 - 20);  // for debugging only: shift in y-direction, TODO: remove!
    }

    void Matcher2::step() {

        // reset root mean squared error, reset parameter update
        aam::Scalar rms = 0;
        MatrixX deltaParam = MatrixX::Zero(4, 1);

        // for each sample position...
        for (size_t i = 0; i < coords.size(); i++) {

            // get the cartesian coordinates (calculated above)
            RowVector2 pt = coords[i];

            // get the corresponding warped point
            RowVector2 warpedPt = transformShape(currentWarp, pt);

            // get gray values from model and image
            aam::Scalar gModel = model.appearanceMean(i);
            aam::Scalar gImg = image.at<unsigned char>((int)warpedPt(0, 1), (int)warpedPt(0, 0));

            // calculate difference of model and image
            aam::Scalar diff = gImg - gModel;
            rms += diff * diff;

            deltaParam += (grad[i] * globalTrafoJacobians[i]).transpose() * diff;
        }

        // (step 8 in figure 7, AAMs revisited)
        // pre-multiply the inverse hessian (see equation 22 in Matthews et. al, "Active Appearance Models Revisited", IJCV, 2004)
        deltaParam = invHessian * deltaParam * aam::Scalar(0.1);  // update with weight 0.1, TODO: remove this artificial weighting of the update
        //deltaParam = invHessian * deltaParam;

        // get the current warp as 3x3 matrix
        MatrixX currentWarp3x3(3, 3);
        currentWarp3x3.block<3, 2>(0, 0) = currentWarp;
        currentWarp3x3(0, 2) = 0;
        currentWarp3x3(1, 2) = 0;
        currentWarp3x3(2, 2) = 1;

        // get the warp update as 3x3 matrix (derive from deltaParam)
        MatrixX updateWarp3x3(3, 3);
        updateWarp3x3.block<3, 2>(0, 0) = paramsToWarp(deltaParam);
        updateWarp3x3(0, 2) = 0;
        updateWarp3x3(1, 2) = 0;
        updateWarp3x3(2, 2) = 1;

        // use this to exclude rotation and scaling from update
        //updateWarp3x3(0, 0) = 1;
        //updateWarp3x3(1, 1) = 1;
        //updateWarp3x3(0, 1) = 0;
        //updateWarp3x3(1, 0) = 0;

        // update the current warp (step 9 in figure 7, AAMs revisited)
        // switch sequence in multiplication of warp matrices compared to AAMs revisited paper
        // (as we are using row vectors, so vectors would be multiplied from left side)
        currentWarp = (updateWarp3x3.inverse() * currentWarp3x3).block<3, 2>(0, 0);

        // calculate the root mean squared error (should be minimized by this optimization procedure)
        rms = sqrt(rms / coords.size());
        std::cout << "Root Mean Squared Error = " << rms << std::endl;

        std::cout << std::endl << "delta Params 3x3: " << std::endl << updateWarp3x3 << std::endl;

        std::cout << std::endl << "delta Params (4x1): " << std::endl << deltaParam << std::endl;

        std::cout << std::endl << "current warp: " << std::endl << currentWarp << std::endl;
    }

}
