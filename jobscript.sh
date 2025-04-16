#!/bin/bash
#PBS -P volta_pilot
#PBS -j oe
#PBS -N phi2_run
#PBS -q volta_gpu
#PBS -l select=1:ncpus=5:mem=50gb:ngpus=1:mpiprocs=1
#PBS -l walltime=04:00:00

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
