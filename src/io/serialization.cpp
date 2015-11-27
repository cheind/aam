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

#include <aam/io/serialization.h>
#include <aam/io/aam_generated.h>
#include <aam/model.h>
#include <aam/traits.h>
#include <iostream>

namespace aam {
    namespace io {

        flatbuffers::Offset<::aam::io::MatrixX> toFlatbuffers(flatbuffers::FlatBufferBuilder &fbb, Eigen::Ref<::aam::MatrixX const> m)
        {
            AamMatrixTraits<double>::MatrixType md = m.cast<double>();
            

            flatbuffers::Offset<flatbuffers::Vector<double> > od = fbb.CreateVector(md.array().data(), md.array().size());

            MatrixXBuilder mb(fbb);
            mb.add_rows(m.rows());
            mb.add_cols(m.cols());
            mb.add_data(od);

            return mb.Finish();            
        }

        void fromFlatbuffers(const ::aam::io::MatrixX &mfb, aam::MatrixX &m)
        {            
            AamMatrixTraits<double>::ConstMatrixMapType mdmap(mfb.data()->data(), mfb.rows(), mfb.cols(), Eigen::Stride<Eigen::Dynamic, 1>(mfb.cols(), 1));
            m = mdmap.cast<::aam::Scalar>();
        }

        void fromFlatbuffers(const ::aam::io::MatrixX &mfb, aam::RowVectorX &m)
        {
            AamMatrixTraits<double>::ConstMatrixMapType mdmap(mfb.data()->data(), mfb.rows(), mfb.cols(), Eigen::Stride<Eigen::Dynamic, 1>(mfb.cols(), 1));
            m = mdmap.cast<::aam::Scalar>().row(0);
        }

        flatbuffers::Offset<::aam::io::MatrixXi> toFlatbuffers(flatbuffers::FlatBufferBuilder &fbb, Eigen::Ref<::aam::RowVectorXi const> m)
        {
            flatbuffers::Offset<flatbuffers::Vector<int> > od = fbb.CreateVector(m.array().data(), m.array().size());

            MatrixXiBuilder mb(fbb);
            mb.add_rows(m.rows());
            mb.add_cols(m.cols());
            mb.add_data(od);

            return mb.Finish();
        }

        void fromFlatbuffers(const ::aam::io::MatrixXi &mfb, ::aam::RowVectorXi &m)
        {
            AamMatrixTraits<int>::ConstMatrixMapType mdmap(mfb.data()->data(), mfb.rows(), mfb.cols(), Eigen::Stride<Eigen::Dynamic, 1>(mfb.cols(), 1));
            m = mdmap.row(0);
        }


        flatbuffers::Offset<::aam::io::ActiveAppearanceModel> toFlatbuffers(flatbuffers::FlatBufferBuilder &fbb, const ::aam::ActiveAppearanceModel &m)
        {
            auto o1 = toFlatbuffers(fbb, m.shapeMean);
            auto o2 = toFlatbuffers(fbb, m.shapeModes);
            auto o3 = toFlatbuffers(fbb, m.shapeModeWeights);
            auto o4 = toFlatbuffers(fbb, m.triangleIndices);
            auto o5 = toFlatbuffers(fbb, m.barycentricSamplePositions);
            auto o6 = toFlatbuffers(fbb, m.appearanceMean);
            auto o7 = toFlatbuffers(fbb, m.appearanceModes);
            auto o8 = toFlatbuffers(fbb, m.appearanceModeWeights);
            
            ActiveAppearanceModelBuilder aamb(fbb);
            aamb.add_shapeMean(o1);
            aamb.add_shapeModes(o2);
            aamb.add_shapeModeWeights(o3);
            //aamb.add_shapeScaleToTrainingSize(m.shapeScaleToTrainingSize);
            aamb.add_triangleIndices(o4);
            aamb.add_barycentricSamplePositions(o5);
            aamb.add_appearanceMean(o6);
            aamb.add_appearanceModes(o7);
            aamb.add_appearanceModeWeights(o8);

            return aamb.Finish();
        }

        void fromFlatbuffers(const ::aam::io::ActiveAppearanceModel &m, ::aam::ActiveAppearanceModel &am)
        {
            fromFlatbuffers(*m.shapeMean(), am.shapeMean);
            fromFlatbuffers(*m.shapeModes(), am.shapeModes);
            fromFlatbuffers(*m.shapeModeWeights(), am.shapeModeWeights);

            //am.shapeScaleToTrainingSize = aam::Scalar(m.shapeScaleToTrainingSize());
            
            fromFlatbuffers(*m.triangleIndices(), am.triangleIndices);
            fromFlatbuffers(*m.barycentricSamplePositions(), am.barycentricSamplePositions);

            fromFlatbuffers(*m.appearanceMean(), am.appearanceMean);
            fromFlatbuffers(*m.appearanceModes(), am.appearanceModes);
            fromFlatbuffers(*m.appearanceModeWeights(), am.appearanceModeWeights);
        }

    }
}