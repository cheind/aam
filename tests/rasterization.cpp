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
#include <aam/rasterization.h>
#include <aam/barycentrics.h>
#include <iostream>

TEST_CASE("rasterization")
{
    // Single triangle
    aam::MatrixX points(1, 3 * 2);
    points << 1.f, 1.f, 3.f, 1.f, 3.f, 3.f;    

    aam::RowVectorXi triangleIds(3);
    triangleIds << 0, 1, 2;

    aam::ParametrizedTriangle pt(points.row(0).segment(0, 2), points.row(0).segment(2, 2), points.row(0).segment(4, 2));

    aam::MatrixX r = aam::rasterizeShape(points, triangleIds, 4, 4, 1);

    REQUIRE(r.rows() == 3);
    REQUIRE(r.cols() == 3);

    REQUIRE(pt.pointAt(r.rightCols(2).row(0)).isApprox(aam::RowVector2(1.5f, 1.5f)));
    REQUIRE(pt.pointAt(r.rightCols(2).row(1)).isApprox(aam::RowVector2(2.5f, 1.5f)));
    REQUIRE(pt.pointAt(r.rightCols(2).row(2)).isApprox(aam::RowVector2(2.5f, 2.5f)));
    REQUIRE((r.leftCols(0).array() == aam::Scalar(0)).all());
}