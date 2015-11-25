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


#include <aam/delaunay.h>
#include <aam/map.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

namespace aam {

    RowVectorXi findDelaunayTriangulation(Eigen::Ref<const RowVectorX> ileavedPoints)
    {
        MatrixX points = fromInterleaved<Scalar>(ileavedPoints);
        eigen_assert(points.cols() == 2);

        // Find min max.
        RowVector2 minC = points.colwise().minCoeff();
        RowVector2 maxC = points.colwise().maxCoeff();

        cv::Rect_<float> bounds(std::floor(minC.x()), std::floor(minC.y()), std::ceil(maxC.x() - minC.x()), std::ceil(maxC.y() - minC.y()));

        cv::Subdiv2D subdiv(bounds);

        std::vector<cv::Point2f> controlPoints;

        for (MatrixX::Index i = 0; i < points.rows(); ++i) {
            cv::Point2f c(points(i, 0), points(i, 1));
            subdiv.insert(c);
            controlPoints.push_back(c);
        }

        std::vector<cv::Vec6f> triangleList;
        subdiv.getTriangleList(triangleList);
        
        RowVectorXi triangleIds(triangleList.size() * 3);

        int validTris = 0;
        for (size_t i = 0; i < triangleList.size(); i++)
        {
            cv::Vec6f t = triangleList[i];

            cv::Point2f p0(t[0], t[1]);
            cv::Point2f p1(t[2], t[3]);
            cv::Point2f p2(t[4], t[5]);

            if (bounds.contains(p0) && bounds.contains(p1) && bounds.contains(p2)) {

                auto iter0 = std::find(controlPoints.begin(), controlPoints.end(), p0);
                auto iter1 = std::find(controlPoints.begin(), controlPoints.end(), p1);
                auto iter2 = std::find(controlPoints.begin(), controlPoints.end(), p2);

                triangleIds(validTris * 3 + 0) = (int)std::distance(controlPoints.begin(), iter0);
                triangleIds(validTris * 3 + 1) = (int)std::distance(controlPoints.begin(), iter1);
                triangleIds(validTris * 3 + 2) = (int)std::distance(controlPoints.begin(), iter2);

                ++validTris;
            }
        }

        return triangleIds.leftCols(validTris * 3);
    }
    
}