# Copy the following into your bashrc or zshrc

```
cvpods_env (){
     export CVPODS_OUTPUT=/group/syzhang/gcluster/outputs/cvpods/
     export CVPODS_HOME=/group/syzhang/gcluster/projects/cvpods/
     export GLOO_SOCKET_IFNAME=ib0
     export NCCL_SOCKET_IFNAME=ib0
}
```
Change the home and output to your own folder

# Example
```
cd /group/syzhang/gcluster/projects/cvpods/playground/detection/coco/rcnn/faster_rcnn.res50.fpn.coco.multiscale.1x
```

# Single Machine Run

```
pods_train --num-gpus 4
```

# Multiple Machine Run

For example, we want train the model on 2 machines:
```
# the first machine
pods_train --num-gpus 4 --dist-url auto --num-machines 2 --machine-rank 0
# the second machine
pods_train --num-gpus 4 --dist-url auto --num-machines 2 --machine-rank 1
```

# GPU check
If you want check the machine with valid gpus
```
pods_train --num-gpus 4 --gpu-check
```

# Q&A

- Import ipdb in anywhere in your code will cause the multi-process initialization error, try pdb when you debug in multi-process mode.