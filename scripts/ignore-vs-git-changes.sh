#!/bin/bash

# Copyright 2026 Ankur Sinha
# Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
# File : scripts/ignore-vs-git-changes.sh
#
# Toggle whether vector stores changes should be tracked by git.
# They change each time they are accessed, but we don't need to store all these
# changes and it's a pain to keep stashing them, each time pre-commit runs, for
# example
#

ignore () {
    pushd vector-stores || exit 1
        find . -type f -print -execdir git update-index --assume-unchanged '{}' \;
    popd || exit 1
    echo > "VECTOR_STORES_IGNORED"

}

unignore () {
    pushd vector-stores || exit 1
        find . -type f -print -execdir git update-index --no-assume-unchanged '{}' \;
    popd || exit 1
    rm -f "VECTOR_STORES_IGNORED"
}

if [ "$#" -ne 1 ]
then
    echo "Only one argument allowed: i/u"
fi

if  [ "$1" == "i" ]
then
    ignore
    git status -s
elif [ "$1" == "u" ]
then
    unignore
    git status -s
else
    echo "Only one argument allowed: i/u"
fi
