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


#include <aam/trainer.h>
#include <aam/model.h>
#include <aam/delaunay.h>
#include <aam/procrustes.h>
#include <aam/pca.h>
#include <aam/rasterization.h>
#include <aam/map.h>

namespace aam {
    
    Trainer::Trainer(const TrainingSet& trainingSet) 
        :_ts(trainingSet)
    {}

    /** shift centroid to origin and scale to 0/1 */
    void Trainer::normalizeShape(Eigen::Ref<MatrixX> shape, Eigen::Ref<RowVectorX> weights, Scalar& scaling) const {
        // Convert points from interleaved to x,y per row.
        MatrixX points = fromInterleaved<Scalar>(shape);

        // Center data
        RowVector2 mean = points.colwise().mean();
        points.rowwise() -= mean;

        // Scale to unit
        RowVector2 minC = points.colwise().minCoeff();
        RowVector2 maxC = points.colwise().maxCoeff();
        RowVector2 dia = maxC - minC;
        scaling = dia.maxCoeff();
        points *= Scalar(1) / scaling;
        weights *= Scalar(1) / scaling;

        shape = toInterleaved<Scalar>(points);
    }

    void Trainer::train(ActiveAppearanceModel& model) {

        aam::Scalar distance = aam::generalizedProcrustes(aam::toEigenHeader<aam::Scalar>(_ts.shapes), 10);

        aam::computePCA(aam::toEigenHeader<aam::Scalar>(_ts.shapes), model.shapeMean, model.shapeModes, model.shapeModeWeights);

        // shape auf 0/1 normalisieren
        normalizeShape(model.shapeMean, model.shapeModeWeights, model.shapeScaleToTrainingSize);

        // (delaunay -> do this in training set)

        //aam::rasterizeShape(); -> bary

        //aam:readShapeImage() pro Training Image -> texture vectors

        //aam::pca(txtur)

        //im model: show model instance for given model parameters

    }

    void Trainer::createTriangulation(TrainingSet& trainingSet) {

        ActiveAppearanceModel model;

        aam::Scalar distance = aam::generalizedProcrustes(aam::toEigenHeader<aam::Scalar>(trainingSet.shapes), 10);

        aam::computePCA(aam::toEigenHeader<aam::Scalar>(trainingSet.shapes), model.shapeMean, model.shapeModes, model.shapeModeWeights);

        trainingSet.triangles = aam::findDelaunayTriangulation(model.shapeMean);
    }

}