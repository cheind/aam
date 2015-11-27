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


#include <aam/model.h>
#include <aam/io/serialization.h>
#include <fstream>

namespace aam {

    bool ActiveAppearanceModel::save(const char *path) const
    {
        flatbuffers::FlatBufferBuilder fbb;
        flatbuffers::Offset<aam::io::ActiveAppearanceModel> oroot = aam::io::toFlatbuffers(fbb, *this);
        fbb.Finish(oroot);

        FILE *f = fopen(path, "wb");
        if (f == 0)
            return false;

        size_t written = fwrite(fbb.GetBufferPointer(), 1, fbb.GetSize(), f);

        fclose(f);

        return written == fbb.GetSize();        
    }

    
    bool ActiveAppearanceModel::load(const char *path)
    {
        FILE *f = fopen(path, "rb");
        if (f == 0)
            return false;

        fseek(f, 0, SEEK_END);
        long fsize = ftell(f);
        fseek(f, 0, SEEK_SET);

        char *buffer = new char[fsize + 1];
        fread(buffer, fsize, 1, f);        
        fclose(f);

        const aam::io::ActiveAppearanceModel *am = aam::io::GetActiveAppearanceModel(buffer);
        aam::io::fromFlatbuffers(*am, *this);

        return true;
    }

}