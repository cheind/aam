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


#include <aam/rasterization.h>
#include <aam/barycentrics.h>
#include <aam/map.h>
#include <aam/views.h>
#include <aam/bilinear.h>
#include <opencv2/opencv.hpp>
#include <iostream>

namespace aam {


    MatrixX rasterizeShape(
        Eigen::Ref<const RowVectorX> pointsInterleaved,
        Eigen::Ref<const RowVectorXi> triangleIds,
        MatrixX::Index imageWidth, MatrixX::Index imageHeight)
    {
        auto points = toSeparatedViewConst<Scalar>(pointsInterleaved);
        MatrixX::Index nTriangles = triangleIds.size() / 3;

        std::vector<RowVector3> coords;

        aam::Scalar minX = points.row(triangleIds(0)).x();
        aam::Scalar maxX = points.row(triangleIds(0)).x();
        aam::Scalar minY = points.row(triangleIds(0)).y();
        aam::Scalar maxY = points.row(triangleIds(0)).y();
        for (MatrixX::Index tri = 0; tri < nTriangles * 3; ++tri) {
            auto p = points.row(triangleIds(tri));
            minX = std::min(p.x(), minX);
            minY = std::min(p.y(), minY);
            maxX = std::max(p.x(), maxX);
            maxY = std::max(p.y(), maxY);
        }

        for (MatrixX::Index tri = 0; tri < nTriangles; ++tri) {
            auto p0 = points.row(triangleIds(tri * 3 + 0));
            auto p1 = points.row(triangleIds(tri * 3 + 1));
            auto p2 = points.row(triangleIds(tri * 3 + 2));

            ParametrizedTriangle pt(p0, p1, p2);

            for (MatrixX::Index y = (MatrixX::Index)minY; y < (MatrixX::Index)(maxY+1); ++y) {
                for (MatrixX::Index x = (MatrixX::Index)minX; x < (MatrixX::Index)(maxX+1); ++x) {
                    RowVector2 p((x + Scalar(0.5)), (y + Scalar(0.5)));
                    RowVector2 bary = pt.baryAt(p);

                    if (pt.isBaryInside(bary)) {
                        coords.push_back(RowVector3(aam::Scalar(tri), bary(0), bary(1)));
                    }
                }
            }
        }

        MatrixX result(coords.size(), 3);

        for (size_t i = 0; i < coords.size(); ++i) {
            result.row(i) = coords[i];
        }
        
        return result;
        
    }
    
    void writeShapeImage(
        Eigen::Ref<const RowVectorX> shape,
        Eigen::Ref<const RowVectorXi> triangleIds,
        Eigen::Ref<const MatrixX> barycentricSamplePositions,
        cv::InputArray colorsAtSamplePositions_,
        cv::InputOutputArray dst_)
    {
        cv::Mat colors = colorsAtSamplePositions_.getMat();
        cv::Mat dst = dst_.getMat();
        
        IplImage colorsipl = colors;
        IplImage dstipl = dst;
        
        // Loop over sample positions and write colors
        
        int triIdLast = -1;
        ParametrizedTriangle pt;
        for (MatrixX::Index i = 0; i < barycentricSamplePositions.rows(); ++i) {
            auto rb = barycentricSamplePositions.row(i);

            int triId = (int)rb(0);
            if (triId != triIdLast) {
                pt.updateVertices(
                    shape.segment(2 * triangleIds(triId * 3 + 0), 2),
                    shape.segment(2 * triangleIds(triId * 3 + 1), 2),
                    shape.segment(2 * triangleIds(triId * 3 + 2), 2));
                triIdLast = triId;
            }
            
            auto p = pt.pointAt(rb.rightCols(2));
            auto pi = (p - RowVector2::Constant(Scalar(0.5))).cast<MatrixX::Index>();
            
            if ((pi.array() >= 0).all() && pi(0) < dst.cols && pi(1) < dst.rows) {
                cvSet2D(&dstipl, pi(1), pi(0), cvGet2D(&colorsipl, i, 0));
            }
        }
    }

    void readShapeImage(
        Eigen::Ref<const RowVectorX> shape,
        Eigen::Ref<const RowVectorXi> triangleIds,
        Eigen::Ref<const MatrixX> barycentricSamplePositions,
        cv::InputArray img_,
        cv::InputOutputArray dst_)
    {
        dst_.create(barycentricSamplePositions.rows(), 1, img_.type());

        cv::Mat dst = dst_.getMat();
        cv::Mat img = img_.getMat();
        
        IplImage dstipl = dst;

        int triIdLast = -1;
        ParametrizedTriangle pt;
        for (MatrixX::Index i = 0; i < barycentricSamplePositions.rows(); ++i) {
            auto rb = barycentricSamplePositions.row(i);

            int triId = (int)rb(0);
            if (triId != triIdLast) {
                pt.updateVertices(
                    shape.segment(2 * triangleIds(triId * 3+0), 2),
                    shape.segment(2 * triangleIds(triId * 3+1), 2),
                    shape.segment(2 * triangleIds(triId * 3+2), 2));
                triIdLast = triId;
            }

            auto p = pt.pointAt(rb.rightCols(2));

            cvSet2D(&dstipl, i, 0, bilinear(img, p(1), p(0)));
        }
    }

    void barycentricToCartesian(
        Eigen::Ref<const RowVectorX> shape,
        Eigen::Ref<const RowVectorXi> triangleIds,
        Eigen::Ref<const MatrixX> barycentricPoints,
        std::vector<RowVector2>& cartesianPoints) 
    {
        // just to be sure: clear the vector of points
        cartesianPoints.clear();

        // Loop over sample positions and get coordinates
        
        int triIdLast = -1;
        ParametrizedTriangle pt;
        for (MatrixX::Index i = 0; i < barycentricPoints.rows(); ++i) {
            auto rb = barycentricPoints.row(i);

            int triId = (int)rb(0);
            if (triId != triIdLast) {
                pt.updateVertices(
                    shape.segment(2 * triangleIds(triId * 3 + 0), 2),
                    shape.segment(2 * triangleIds(triId * 3 + 1), 2),
                    shape.segment(2 * triangleIds(triId * 3 + 2), 2));
                triIdLast = triId;
            }
            
            auto p = pt.pointAt(rb.rightCols(2));
            cartesianPoints.push_back(p);
        }
    }
    
}