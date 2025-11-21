# Usage on HPC
## Environment
`module load conda`

`conda env create -f environment.yml`

`conda activate partfield`

`pip install huggingface_hub`
## Download pretrained model

`mkdir model`

`cd model`

`hf download mikaelaangel/partfield-ckpt`
Make sure the model is in `/model`
## Submit and track a job
In the terminal, use the following command, where `BATCH_ID` should be specified.

`sh pipeline_infere_auto.sh BATCH_ID`

The pipeline include downloading a batch of data and infere. The data directory and output directory are listed in `pipeline_infere.sh`. Please change them accordingly. `SCRATCH_BASE="/scratch/eecs442f25_class_root/eecs442f25_class/jonzhe"` -> `SCRATCH_BASE="/scratch/eecs442f25_class_root/eecs442f25_class/YOUR_USER_NAME"`

You can see the job running status by `squeue -u $USER`. 

The output log will be `partfield_pipeline_${BATCH_ID}.log` in the current directory.
