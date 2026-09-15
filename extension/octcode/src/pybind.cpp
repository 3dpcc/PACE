#include "include/octree.h"

namespace py = pybind11;

PYBIND11_MODULE(octcode, m) {
    m.doc() = "octcode - Morton-code based octree";

    py::class_<OctreeEnc, std::shared_ptr<OctreeEnc>>(m, "OctreeEnc")
        .def(py::init<int32_t>(), py::arg("max_depth"))
        .def("build",    &OctreeEnc::build,    "Build octree from point cloud", py::arg("pc"))
        .def("traverse", &OctreeEnc::traverse, "Traverse octree and return context array");

    py::class_<OctreeDec, std::shared_ptr<OctreeDec>>(m, "OctreeDec")
        .def(py::init<int32_t>(), py::arg("max_depth"))
        .def("calc_context",  &OctreeDec::calc_context,  "Calculate context for the given level", py::arg("depth"))
        .def("calc_children", &OctreeDec::calc_children, "Expand children from predicted symbols", py::arg("syms"), py::arg("depth"))
        .def("get_context",   &OctreeDec::get_context,   "Get context array for the given level", py::arg("depth"))
        .def("get_coords",    &OctreeDec::get_coords,    "Get node origin coords for the given level", py::arg("depth"));

    m.def("build_octree",
        [](py::array_t<int32_t>& pc, int32_t depth) {
            auto octree = std::make_shared<OctreeEnc>(depth);
            octree->build(pc);
            return octree->traverse();
        },
        "Convenience: build octree and return context",
        py::arg("pc"), py::arg("depth")
    );
}
