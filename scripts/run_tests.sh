#!/bin/bash

# Copyright 2026 Ankur Sinha
# Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
# File : test.sh
#
# Scripts to run tests in all packages.


set -e

echo ">> Unit tests"
for d in *_pkg
do
    if [ -d "${d}/tests" ]
    then
        pushd "$d" || exit 1
        pytest -v -n auto
        popd || exit 1
    fi
done
