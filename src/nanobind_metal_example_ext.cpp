#include <nanobind/nanobind.h>

#include "metal_add.h"
#include "metal_utils.h"

namespace nb = nanobind;


NB_MODULE(nanobind_metal_example_ext, m) {
    m.doc() = "This is a \"hello world\" example with nanobind and Metal";
    m.def("vecf_add", vecf_add);
    m.def("vecf_add_out", vecf_add_out);
    m.def("synchronize", metal_synchronize);
}
