Most of the requirements of this projects are exactly the same as [cvpods](https://github.com/Megvii-BaseDetection/cvpods). If you have any problem of your environment, you should check their [issues page](https://github.com/Megvii-BaseDetection/cvpods/issues) first. Hope you will find the answer.


# Install Packages

## Conda Environments

``` bash
# first, make sure that your conda is setup properly with the right environment
# for that, check that `which conda`, `which pip` and `which python` points to the
# right path. From a clean conda env, this is what you need to do

conda create -y --name sgtr
conda activate sgtr
```

## MISC packages
```
pip install -r requirements.txt
```

## Pytorch 1.10

``` bash
# CUDA 10.2
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=10.2 -c pytorch
# CUDA 11.3
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
```

## Path Environments
- Add the path env into your bashrc or zshrc.

``` bash
cvpods_env (){
     export CVPODS_OUTPUT=/path/to/your/outputs/
     export CVPODS_HOME=/path/to/your/outputs/
     export GLOO_SOCKET_IFNAME=ib0
     export NCCL_SOCKET_IFNAME=ib0
}
cvpods_env
```
- Change the /path/to/your/outputs/ to your own folder for saving the training outputs.

# Build Project
```
python setup.py build develop
```

# Update VG Evaluation Config
Update `/your/project/dir/` in the L528 of `cvpods/evaluation/sgg_vg_evaluation.py`
``` python
def classic_vg_sgg_evaluation(
        cfg,
        predictions,
        groundtruths,
        predicates_categories: list,
        output_folder,
        logger,
):
    # get zeroshot triplet
    zeroshot_triplet = torch.load(
        "/your/project/dir/sgtr_release/datasets/vg/vg_motif_anno/zeroshot_triplet.pytorch",
        map_location=torch.device("cpu")).long().numpy()

```