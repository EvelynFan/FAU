## FAU

Implementation of the paper:

**Facial Action Unit Intensity Estimation via Semantic Correspondence Learning with Dynamic Graph Convolution**. Yingruo Fan, Jacqueline C.K. Lam and Victor O.K. Li.  ***AAAI 2020*** [[PDF]](https://aaai.org/Papers/AAAI/2020GB/AAAI-FanY.6827.pdf)

## Overview

<p align="center">
<img src="examples/framework.jpg" width="88%" />
</p>

## Environment
- Ubuntu 18.04.4
- Python 3.7
- Tensorflow 1.14.0

## Dependencies
Check the packages needed or simply run the command
```console
❱❱❱ pip install -r requirements.txt
```

***Preparation***

For data preparation, please make a request for the [BP4D database](http://www.cs.binghamton.edu/~lijun/Research/3DFE/3DFE_Analysis.html) and the [DISFA database](http://mohammadmahoor.com/disfa/). For annotation files, you need to convert them into json format.

The backbone model is initialized from the pretrained [ResNet-V1-50](https://github.com/tensorflow/models/tree/master/research/slim). Please download it under `${DATA_ROOT}`. You can change default path by modifying `config.py`.

***Demo***

Download the pretrained model from [GoogleDrive]() and put it under `${DATA_ROOT}/output/models/`.

```console
❱❱❱ python demo.py --gpu 1 --epoch 10 
```
Then, the visualized heatmaps will be generated in the `vis_dir` folder. 

***Training***
```console
❱❱❱ python train.py --gpu 1
```
***Training***
```console
❱❱❱ python test.py --gpu 1 --epoch 10
```

## Citation

    @inproceedings{fan2020fau,
        title = {Facial Action Unit Intensity Estimation via Semantic Correspondence Learning with Dynamic Graph Convolution},
        author = {Fan, Yingruo and Lam, Jacqueline and Li, Victor},
        booktitle = {Thirty-Fourth AAAI Conference on Artificial Intelligence},
        year={2018}
    }
