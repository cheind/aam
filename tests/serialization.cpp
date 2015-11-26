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
#include <aam/aam.h>
#include <iostream>

TEST_CASE("serialize")
{
    aam::MatrixX m = aam::MatrixX::Random(5, 10);

    aam::RowVectorXi tris(3);
    tris << 0, 1, 2;
    
    aam::ActiveAppearanceModel am;
    
    am.appearanceMean = m.row(0);
    am.appearanceModes = m.topRows(2);
    am.appearanceModeWeights = m.row(1);
    am.barycentricSamplePositions = m.topRows(3);
    am.shapeMean = m.row(3);
    am.shapeModes = m.topRows(4);
    am.shapeModeWeights = m.row(4);
    am.triangleIndices = tris;
    am.shapeScaleToTrainingSize = 10;

    am.save("aam.bin");

    // Clear appearance model and re-load
    am = aam::ActiveAppearanceModel();
    am.load("aam.bin");

    REQUIRE(am.appearanceMean.isApprox(m.row(0)));
    REQUIRE(am.appearanceModes.isApprox(m.topRows(2)));
    REQUIRE(am.appearanceModeWeights.isApprox(m.row(1)));
    REQUIRE(am.barycentricSamplePositions.isApprox(m.topRows(3)));
    REQUIRE(am.shapeMean.isApprox(m.row(3)));
    REQUIRE(am.shapeModes.isApprox(m.topRows(4)));
    REQUIRE(am.shapeModeWeights.isApprox(m.row(4)));
    REQUIRE(am.triangleIndices.isApprox(tris));
    REQUIRE(am.shapeScaleToTrainingSize == 10);
}