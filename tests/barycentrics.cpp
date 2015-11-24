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
#include <aam/barycentrics.h>
#include <iostream>


TEST_CASE("barycentrics")
{
    aam::ParametrizedTriangle pt;
    
    pt.updateVertices(aam::RowVector2(1,1), aam::RowVector2(3,1), aam::RowVector2(2,4));
    
    // Corner tests
    REQUIRE(pt.baryAt(aam::RowVector2(1,1)).isApprox(aam::RowVector2(0,0)));
    REQUIRE(pt.baryAt(aam::RowVector2(3,1)).isApprox(aam::RowVector2(1,0)));
    REQUIRE(pt.baryAt(aam::RowVector2(2,4)).isApprox(aam::RowVector2(0,1)));
    
    REQUIRE(!pt.isPointInside(aam::RowVector2(0,0)));
    REQUIRE(!pt.isPointInside(aam::RowVector2(1,0)));
    REQUIRE(!pt.isPointInside(aam::RowVector2(0,1)));
    REQUIRE(!pt.isPointInside(aam::RowVector2(0.5,0.5)));
    REQUIRE(pt.isPointInside(aam::RowVector2(1,1)));
    REQUIRE(pt.isPointInside(aam::RowVector2(2,2)));
    
    // Forward backward tests
    aam::MatrixX barys = aam::MatrixX::Random(100, 2);
    barys += aam::MatrixX::Constant(100, 2, 1.f);
    barys /= 2.f;
    
    for (int i = 0; i < 100; ++i) {
        auto r = barys.row(i);
        bool shouldbeinside = r.sum() <= aam::Scalar(1);
        
        aam::RowVector2 b = pt.baryAt(pt.pointAt(r));
        REQUIRE(b.isApprox(r));
        if (shouldbeinside) {
            REQUIRE(pt.isBaryInside(b));
        }
    }
    
}