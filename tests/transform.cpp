/**
This file is part of Active Appearance Models (AMM).

Copyright Christoph Heindl 2015
Copyright Sebastian Zambal 2015

AMM is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

AMM is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with AMM.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "catch.hpp"
#include <aam/transform.h>
#include <Eigen/Geometry>
#include <iostream>

TEST_CASE("transform")
{
    aam::MatrixX shapes(2, 4);
    shapes << 1, 2, 3, 4, 
              5, 6, 7, 8;
    
    aam::MatrixX shapesResult(2, 4);
    shapesResult << 2.5, 4.5, 6.5, 8.5,
              10.5, 12.5, 14.5, 16.5;


    // Test with affine transforms
    Eigen::Transform<aam::Scalar, 2, Eigen::AffineCompact> t;
    t = Eigen::Translation<aam::Scalar, 2>(0.5, 0.5) * Eigen::Scaling(aam::Scalar(2));
    
    aam::Affine2 m = t.matrix().transpose();
    
    
    // Version 1
    // Can work in place (rather slow) but output needs to be pre-allocated.
    aam::RowVectorX r1(4);
    aam::transformShape(m, shapes.row(0), r1);
    REQUIRE(r1.isApprox(shapesResult.row(0)));
    
    // Version 2
    // In place
    aam::RowVectorX r2 = shapes.row(1);
    aam::transformShapeInPlace(m, r2);
    REQUIRE(r2.isApprox(shapesResult.row(1)));
    
    
    // Version 3
    // Return transformed
    aam::RowVectorX r3 = aam::transformShape(m, shapes.row(0));
    REQUIRE(r3.isApprox(shapesResult.row(0)));
}