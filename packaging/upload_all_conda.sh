#!/usr/bin/bash

set -e

# This is fvcore's API token to anaconda-cloud
TOKEN=${PTV_CONDA_TOKEN:?NO_TOKEN}

ls -Rl packaging/output_files

if [[ -f SKIP ]]
then
   echo NOTHING TO DO
else
    for file in packaging/output_files/linux-64/*.bz2
    do
        anaconda --verbose -t ${PTV_CONDA_TOKEN} upload --force ${file}
    done
fi