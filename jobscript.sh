#!/bin/bash
#PBS -q volta_gpu
#PBS -N watermarp_paraphrase
#PBS -l select=1:ncpus=5:ngpus=1:mem=100gb
#PBS -l walltime=01:00:00
#PBS -j oe

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
