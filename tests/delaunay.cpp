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
#include <aam/delaunay.h>

TEST_CASE("delaunay")
{
    // Single triangle
    aam::MatrixX points(1, 3 * 2);
    points << 1.f, 1.f, 3.f, 1.f, 2.f, 4.f;

    aam::RowVectorXi triangleIds = aam::findDelaunayTriangulation(points);

    REQUIRE(triangleIds.size() == 3);
    REQUIRE(triangleIds(0) == 0);
    REQUIRE(triangleIds(1) == 1);
    REQUIRE(triangleIds(2) == 2);
    
}