#!/usr/bin/env bash
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

set -e

if [ -z "$HERO_INSTALL" ]; then
  echo "Fatal: 'HERO_INSTALL' environment variable is not set!"
  exit 1
fi

if [ -z "$CMUX_ROOT" ]; then
  echo "Fatal: 'CMUX_ROOT' environment variable is not set!"
  exit 1
fi

cd "$ROOT"
make tc-pulp
make tc-har-olinux
make sdk-pulp
make sdk-har
make tc-llvm

cd "$CMUX_ROOT"
make cmux
cd src/pulp
HERO_ROOT="$ROOT" ./build-pulp-lib.sh

cd "$ROOT"
make br-har-exilzcu102

cp "$CMUX_ROOT/lib/libpremnotify-cpu.so" "$HERO_INSTALL/../output/br-har-exilzcu102/target/usr/lib/"
