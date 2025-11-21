#!/bin/bash

BATCH_ID=$1
sbatch --output=partfield_pipeline_${BATCH_ID}.log pipeline_infere.sh $BATCH_ID