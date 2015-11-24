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

#ifndef AAM_BARYCENTRICS_H
#define AAM_BARYCENTRICS_H

#include <aam/types.h>

namespace aam {
    
    /** A bary-centric coordinates parametrized triangle */
    class ParametrizedTriangle {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        
        /** Empty constructor */
        inline ParametrizedTriangle()
        {}
        
        /** Init with triangle vertices */
        inline ParametrizedTriangle(const RowVector2 &a, const RowVector2 &b, const RowVector2 &c)
        {
            updateVertices(a, b, c);
        }
        
        /** Update vertices */
        inline void updateVertices(const RowVector2 &a, const RowVector2 &b, const RowVector2 &c)
        {
            _a = a;
            _b = b;
            _c = c;
            
            _basis.row(0) = b - a;
            _basis.row(1) = c - a;
            _denom = Scalar(1) / ((_b.x() - _a.x())*(_c.y() - _a.y()) - (_b.y() - _a.y())*(_c.x() - _a.x()));
            
        }
        
        /** Given bary-centric coordinates retrieve the cartesian coordinates. */
        inline RowVector2 pointAt(const RowVector2 &t) const {
            return _a + t * _basis;
        }
        
        /** Given cartesian coordinates return bary-centric coordinates. */
        inline RowVector2 baryAt(const RowVector2 &t) const {
            RowVector2 bary;
            bary(0) = ((t.x() - _a.x())*(_c.y() - _a.y()) - (t.y() - _a.y())*(_c.x() - _a.x())) * _denom;
            bary(1) = ((t.y() - _a.y())*(_b.x() - _a.x()) - (t.x() - _a.x())*(_b.y() - _a.y())) * _denom;
            return bary;
        }
        
        /** Test if cartesian point is inside of the triangle. */
        inline bool isPointInside(const RowVector2 &p) const {
            return isBaryInside(baryAt(p));
        }
        
        /** Test parametrization represents a point inside of the triangle. */
        inline bool isBaryInside(const RowVector2 &bary) const {
            return (bary.array() >= aam::Scalar(0)).all() && bary.sum() <= 1.f;
        }
    
        
    private:
        Matrix2 _basis;
        RowVector2 _a, _b, _c;
        Scalar _denom;
    };

}

#endif