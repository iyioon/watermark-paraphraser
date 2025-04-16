#!/bin/bash
#PBS -P volta_pilot
#PBS -j oe
#PBS -N watermark_paraphrase
#PBS -q volta_gpu
#PBS -l select=1:ncpus=5:mem=50gb:ngpus=1:mpiprocs=1
#PBS -l walltime=24:00:00

cd $PBS_O_WORKDIR;
np=$(cat ${PBS_NODEFILE} | wc -l);

image="/app1/common/singularity-img/3.0.0/tensorflow_1.12_nvcr_19.01-py3.simg"

singularity exec $image bash << EOF > stdout.$PBS_JOBID 2> stderr.$PBS_JOBID

# Test print
echo "This is a test print"

# you can put more commands here
# echo “Hello World”
EOF
