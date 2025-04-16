#!/bin/bash
#PBS -P watermarkparaphrase
#PBS -j oe
#PBS -N watermarkparaphrase
#PBS -l select=1:ncpus=1
#PBS -l place=free:shared

cd ${PBS_O_WORKDIR}

# Create a temporary scratch directory
scratch=/scratch/${USER}/${PBS_JOBID}
export TMPDIR=$scratch
mkdir -p $scratch

# Run Python program:
python demo/generate.py data/in/sample.txt --model microsoft/phi-2 --key 42 --output data/out/sample_paraphrased.text --verbose

# Clean up scratch space
rm -rf $scratch
exit $?
