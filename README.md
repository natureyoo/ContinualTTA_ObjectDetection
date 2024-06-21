# Continual Test-Time Object Detection

## Installation Instruction
We use Python 3.10, PyTorch 1.11.0 (CUDA 11.8 build).
The codebase is built on [Detectron2](https://github.com/facebookresearch/detectron2).

```angular2
conda create -n cta_od python=3.10

Conda activate cta_od

conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.8 -c pytorch

cd ContinualTTA_ObjectDetection
pip install -r requirements.txt

## Make sure you have GCC and G++ version <=8.0
cd ..
python -m pip install -e ContinualTTA_ObjectDetection

```
## Dataset Preparation

Please follow dataset structure below.

1. Coco & coco-corruption
    ```
    - coco
        - train2017
        - val2017
        - val2017-snow
        - val2017-frost
        ...
    ```

2. SHIFT
    ```
    - shift
        - discrete
            - images
                - train
                    - front
                        - images
                            ...
                        - det_2d.json
                        - seq.csv
                - val
                    - front
                        ...
        - continuous1x
        - continuous10x
    ```

## Model Checkpoints and Feature Statistics

- Source pre-trained models and source feature statistics can be downloaded from [Link](https://drive.google.com/drive/folders/17wS8BJrRBjikGkp3mD_8JGt0b4HO6K7M?usp=sharing).


## Citation

If you found IRG SFDA useful in your research, please consider starring ⭐ us on GitHub and citing 📚 us in your research!

```bibtex
@InProceedings{Yoo_2024_CVPR,
    author    = {Yoo, Jayeon and Lee, Dongkwan and Chung, Inseop and Kim, Donghyun and Kwak, Nojun},
    title     = {What How and When Should Object Detectors Update in Continually Changing Test Domains?},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {23354-23363}
}
```

