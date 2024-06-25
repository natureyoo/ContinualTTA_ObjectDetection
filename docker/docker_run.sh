docker run --gpus all -it --ipc=host --name=ttadapt -v $PWD:/workspace -v /home/$USER/data/:/workspace/datasets/ -v /etc/localtime:/etc/localtime pytorch/pytorch:1.9.0-cuda11.1-cudnn8-devel
#pytorch/pytorch:1.9.0-cuda11.1-cudnn8-devel
#nvcr.io/nvidia/pytorch:21.05-py3