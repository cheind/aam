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

#ifndef AAM_MODEL_H
#define AAM_MODEL_H

#include <aam/types.h>

namespace aam {
    
    /** Training active appearance model. */
    class ActiveAppearanceModel {
    public:
        
        /** 1xN Mean shape vector.
            Given in normalized shape coordinates.
            Format:
                x0, y0, x1, y1, ...
         */
        RowVectorX shapeMean;
        
        /** Most dominant shape modes (eigenvectors) in rows starting
            with the least significant mode.
            Format: 
                x0, y0, x1, y1, ...
                x0, y0, x1, y1, ...
         */
        MatrixX shapeModes;

        /** Eigen values corresponding to shape modes
            Format:
                lambda0, lambda1, lambda2, ...
         */
        RowVectorX shapeModeWeights;
        
        /** Scale factor to be applied to normalized coordinates to bring
            coordinates back to training image size. 
         */
        Scalar shapeScaleToTrainingSize;
        
        /** List of triangle indices referencing shape points. 
            Format:
                t0_a, t0_b, t0_c, t1_a, t1_b, t1_c, ... 
         */
        RowVectorXi triangleIndices;
        
        /** Nx3 matrix with bary centric sample positions in rows. These
            sample positions are generated in such a way that they represent
            a dense pixel based sampled when considered in the mean shape and
            training dimensions.
         
            Format:
                triangleId, alpha, beta
                triangleId, alpha, beta
         */
        MatrixX barycentricSamplePositions;
        
        /** 1xN Mean appearance vector.
            Given in normalized shape coordinates.
            Format:
                i0, i1, i2, ...
         */
        RowVectorX appearanceMean;
        
        /** Most dominant appearance modes (eigenvectors) in rows starting
            with the least significant mode.
            Format:
                i0, i1, i2, ...
                i0, i1, i2, ...
         */
        MatrixX appearanceModes;

        /** Eigen values corresponding to appearance modes
            Format:
                lambda0, lambda1, lambda2, ...
        */
        RowVectorX appearanceModeWeights;
        
    };
   
}

#endif