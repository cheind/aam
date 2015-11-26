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

#ifndef AAM_BILINEAR_H
#define AAM_BILINEAR_H

#include <aam/types.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/core_c.h>
#include <opencv2/imgproc/imgproc.hpp>

namespace aam {

    inline cv::Scalar bilinear(const cv::Mat &img, Scalar y, Scalar x)
    {
        x -= aam::Scalar(0.5);
        y -= aam::Scalar(0.5);

        const int ix = static_cast<int>(std::floor(x));
        const int iy = static_cast<int>(std::floor(y));

        int x0 = cv::borderInterpolate(ix, img.cols, cv::BORDER_REFLECT_101);
        int x1 = cv::borderInterpolate(ix + 1, img.cols, cv::BORDER_REFLECT_101);
        int y0 = cv::borderInterpolate(iy, img.rows, cv::BORDER_REFLECT_101);
        int y1 = cv::borderInterpolate(iy + 1, img.rows, cv::BORDER_REFLECT_101);

        Scalar a = x - (Scalar)ix;
        Scalar b = y - (Scalar)iy;
        
        IplImage iplimg = img;

        const cv::Scalar f0 = (cv::Scalar)cvGet2D(&iplimg, y0, x0);
        const cv::Scalar f1 = (cv::Scalar)cvGet2D(&iplimg, y0, x1);
        const cv::Scalar f2 = (cv::Scalar)cvGet2D(&iplimg, y1, x0);
        const cv::Scalar f3 = (cv::Scalar)cvGet2D(&iplimg, y1, x1);
        
        const cv::Vec<double, 4> r = (f0 * (Scalar(1) - a) + f1 * a) * (Scalar(1) - b) +
                                     (f2 * (Scalar(1) - a) + f3 * a) * b;
        
        return cv::Scalar(r[0], r[1], r[2], r[3]);
    }
}

#endif