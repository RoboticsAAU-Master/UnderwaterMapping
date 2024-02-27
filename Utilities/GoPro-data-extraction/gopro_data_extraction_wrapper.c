// main.cpp
#include <pybind11/pybind11.h>
#include "gopro_data_extraction.c"

namespace py = pybind11;

PYBIND11_MODULE(gopro_data_extractor, m)
{
    m.def("extract_metadata", &readMP4File, "Function that processes a GoPro file and extracts the metadata.");
}