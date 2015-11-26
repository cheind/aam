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

#include <aam/aam.h>
#include <aam/trainer.h>
#include <aam/io.h>

#include <opencv2/highgui/highgui.hpp>
#include <iostream>

/**

Main entry point.

*/
int main(int argc, char **argv)
{
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " directory" << std::endl;
        return 0;
    }

    aam::TrainingSet trainingSet;
    aam::loadAsfTrainingSet(argv[1], trainingSet);

    aam::Trainer trainer(trainingSet);
    //trainer.train();


    //aam::showTrainingSet(trainingSet);

    return 0;
}




