nanobind_gpu_example
================

This repository contains a tiny project showing how to create C++ bindings
using [nanobind](https://github.com/wjakob/nanobind) and
[scikit-build-core](https://scikit-build-core.readthedocs.io/en/latest/index.html),
that utilize GPU (currently Metal or CUDA) compute shaders. It was derived from the
original [nanobind example project](https://github.com/wjakob/nanobind_example)
developed by [@wjakob](https://github.com/wjakob).

Installation
------------

1. Clone this repository; `cd nanobind_gpu_example`
2. If running on Apple Silicon, copy [metal-cpp](https://developer.apple.com/metal/cpp/) into the root of this project directory.
3. Run `pip install .`

Afterwards, you should be able to run the test, which profiles Torch MPS versus raw Metal with nanobind and Torch CPU.

```bash
python tests/test_basic.py
```

License
-------

I kept the original [LICENSE](./LICENSE) file from Wenzel's example because
I don't really understand how licenses work and I don't want to get in trouble.
