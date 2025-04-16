#!/bin/bash

#PBS -P myproj
#PBS -j oe
#PBS -N myprog
#PBS -l select=1:ncpus=1
#PBS -l place=free:shared

cd ${PBS_O_WORKDIR};   ## this line is needed, do not delete.

# Create a temporary scratch directory
scratch=/scratch/${USER}/${PBS_JOBID}
export TMPDIR=$scratch
mkdir -p $scratch

# Run Python program:
python demo/generate.py data/in/sample.txt --model microsoft/phi-2 --key 42 --output data/out/sample_paraphrased.text --verbose

# Clean up scratch space if
rm -rf $scratch
exit $?
