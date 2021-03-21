# Copyright (c) 2020 ETH Zurich, University of Bologna
# Licensed under the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0
#
# Authors:
# - Andreas Kurth <akurth@iis.ee.ethz.ch>

source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../common.sh"

cd "$HERO_ROOT/output/$BR_OUTPUT_SUBDIR"
make hero-openmp-dirclean
make hero-openmp-rebuild
make hero-openmp-reinstall

cd "$HERO_ROOT/pulp/sdk"
if defined_or_warning_and_exit \
    PULP_SDK_HOME "Will not build PULP library nor install it in the PULP SDK."; then
  # Execute in subshell to not clutter environment of this shell.
  (
    export PULP_RISCV_GCC_TOOLCHAIN="$HERO_INSTALL"
    source configs/hero-urania.sh
    source configs/platform-hsa.sh
    plpbuild --m libomptarget-pulp-rtl clean build
  )
fi

if defined_or_warning_and_exit \
    HERO_INSTALL "Will not install header and libs on development machine."; then
  chmod -R u+w "$HERO_INSTALL"
  cd "$HERO_ROOT/output/$BR_OUTPUT_SUBDIR"
  cp -v host/aarch64-buildroot-linux-gnu/sysroot/usr/include/omp*.h \
      "$HERO_INSTALL/aarch64-buildroot-linux-gnu/sysroot/usr/include/"
  cp -v host/aarch64-buildroot-linux-gnu/sysroot/usr/lib/libomp* \
      "$HERO_INSTALL/aarch64-buildroot-linux-gnu/sysroot/usr/lib/"
  chmod -R u-w "$HERO_INSTALL"
fi

if defined_or_warning_and_exit HERO_BOARD_HOSTNAME "Will not copy shared library to board."; then
  if defined_or_warning_and_exit HERO_BOARD_LIB_PATH "Will not copy shared library to board."; then
    cd "$HERO_ROOT/output/$BR_OUTPUT_SUBDIR"
    scpv target/usr/lib/libomp* $HERO_BOARD_HOSTNAME:"$HERO_BOARD_LIB_PATH"
  fi
fi
