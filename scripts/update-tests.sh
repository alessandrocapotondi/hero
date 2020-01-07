#!/bin/bash -e

# This is the script you use when you do a local test
# getting tests
git clone git@iis-git.ee.ethz.ch:pulp-tests/ml_tests.git \
    tests/ml_tests
git clone git@iis-git.ee.ethz.ch:pulp-sw/pulp_tests.git \
    tests/pulp_tests
git clone git@iis-git.ee.ethz.ch:pulp-tests/rt-tests.git \
    tests/rt-tests
git clone git@iis-git.ee.ethz.ch:pulp-sw/parallel_bare_tests.git \
    tests/parallel_bare_tests
git clone git@iis-git.ee.ethz.ch:pulp-sw/riscv_tests.git \
    tests/riscv_tests
git clone git@iis-git.ee.ethz.ch:pulp-sw/sequential_bare_tests.git \
    tests/sequential_bare_tests

# using "stable" versions
echo "Using stable versions of tests"
git -C tests/ml_tests checkout -q 481c83b30c2cb7b823d1769b1c0d6a30a3a3b9b0
git -C tests/pulp_tests checkout -q ce367a85b9d4b15e92b57cdfa7b715ecf6a92a45
git -C tests/rt-tests checkout -q c19233304eb58a388c523e55d467f4abc666358f
git -C tests/parallel_bare_tests checkout -q 91b1bad09d088df9140f5391a87df3f6ebfab344
git -C tests/riscv_tests checkout -q 8bbda74844c4baaa86efa89a2e19c985fe4095c3
git -C tests/sequential_bare_tests checkout -q 149f92fe3ecbd82cde87c3865b4d25d8c825a25d
