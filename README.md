# Local Augmentation for Graph Neural Networks

This repository contains experiments done as part of course project for the paper ["Local Augmentation for Graph Neural Networks"](https://openreview.net/pdf?id=HOlhtomacz).

## Dependencies
- CUDA 10.2.89
- python 3.6.8
- pytorch 1.9.0
- pyg 2.0.3

## Usage
- For semi-supervised setting, run the following script
```sh
cd Citation
bash semi.sh
```

- For full-supervised setting, run the following script
```sh
cd OGB
# If you want to pre-train the generative model, run the following command:
python cvae_generate_products.py --latent_size 10 --pretrain_lr 1e-5 --total_iterations 10000 --batch_size 8192
# Train downstream GNNs
bash full.sh
```

## Additions : 
1) Graph Classification : @Aziz-Shameem
   
CVAE Pretrining : Run cvae_train.py with the appropriate parameters  
Model Training : Run lagin_graphlevel.py with the appropriate parameters (after pretraining the CVAE)

```sh
cd Citation
# for pretraining the CVAE
python cvae_train.py
# for training the GIN model
python lagin_graphlevel.py
```
2) Link Prediction: @BhavyaKohli
3) Normalizing-Flow Model: @prateekgargx

For semi-supervised setting, we provide two pre-trained generative models: Conditional VAE, and Conditional Normalizing Flow.
If you want to pre-train yourself, use:

```sh
cd Citation
!python cvae_generate_citation.py --model 1 [other-parameters]
# 0 for CVAE, 1 for CNF
```

## Citation
```
@inproceedings{liu2022local,
  title={Local augmentation for graph neural networks},
  author={Liu, Songtao and Ying, Rex and Dong, Hanze and Li, Lanqing and Xu, Tingyang and Rong, Yu and Zhao, Peilin and Huang, Junzhou and Wu, Dinghao},
  booktitle={International Conference on Machine Learning},
  year={2022}
}
```
