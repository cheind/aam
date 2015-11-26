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

#ifndef AAM_SERIALIZATION_H
#define AAM_SERIALIZATION_H

#include <aam/fwd.h>
#include <aam/types.h>
#include <aam/io/aam_generated.h>

namespace aam {
    namespace io {

        /** Serialize matrix to flatbuffers storage */
        flatbuffers::Offset<::aam::io::MatrixX> toFlatbuffers(flatbuffers::FlatBufferBuilder &fbb, Eigen::Ref<::aam::MatrixX const> m);

        /** Serialize matrix from flatbuffers storage */
        void fromFlatbuffers(const ::aam::io::MatrixX &mfb, aam::MatrixX &m);

        /** Serialize matrix from flatbuffers storage */
        void fromFlatbuffers(const ::aam::io::MatrixX &mfb, aam::RowVectorX &m);

        /** Serialize matrix to flatbuffers storage */
        flatbuffers::Offset<::aam::io::MatrixXi> toFlatbuffers(flatbuffers::FlatBufferBuilder &fbb, Eigen::Ref<::aam::RowVectorXi const> m);

        /** Serialize matrix from flatbuffers storage */
        void fromFlatbuffers(const ::aam::io::MatrixXi &mfb, ::aam::RowVectorXi &m);

        /** Serialize ActiveAppearanceModel to flatbuffers storage */
        flatbuffers::Offset<::aam::io::ActiveAppearanceModel> toFlatbuffers(flatbuffers::FlatBufferBuilder &fbb, const ::aam::ActiveAppearanceModel &m);

        /** Serialize ActiveAppearanceModel from flatbuffers storage */
        void fromFlatbuffers(const ::aam::io::ActiveAppearanceModel &mfb, ::aam::ActiveAppearanceModel &m);
    }    
}

#endif



