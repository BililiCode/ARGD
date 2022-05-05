# Attention Relation Graph Distillation

![Python 3.6](https://img.shields.io/badge/python-3.6-DodgerBlue.svg?style=plastic)
![Pytorch 1.10](https://img.shields.io/badge/pytorch-1.2.0-DodgerBlue.svg?style=plastic)
![CUDA 10.0](https://img.shields.io/badge/cuda-10.0-DodgerBlue.svg?style=plastic)
![License CC BY-NC](https://img.shields.io/badge/license-CC_BY--NC-DodgerBlue.svg?style=plastic)

We have already uploaded the `all2one` pretrained backdoor student model(i.e. gridTrigger WRN-16-1, target label 0) and the clean teacher model(i.e. WRN-16-1) in the path of `./weight/s_net` and `./weight/t_net` respectively. 

For evaluating the performance of  ARGD, you can easily run command:

```bash
$ python main-ARGD.py 
```
where the default parameters are shown in `config.py`.

The trained model will be saved at the path `weight/erasing_net/<s_name>.tar`

Please carefully read the `main.py` and `configs.py`, then change the parameters for your experiment.

### Erasing Results on BadNets under 5% clean data ratio
| Dataset  | Baseline ACC | Baseline ASR | ARGD ACC | ARGD ASR |
| -------- | ------------ | ------------ |  ------- | -------   |
| CIFAR-10 | 80.08        | 100.0        |  79.81   |   2.10    |

---

## Training your own backdoored model
We have provided a `DatasetBD` Class in `data_loader.py` for generating training set of different backdoor attacks. 

For implementing backdoor attack(e.g. GridTrigger attack), you can run the below command:

```bash
$ python train_badnet.py 
```

This command will train the backdoored model and print clean accuracies and attack rate. You can also select the other backdoor triggers reported in the paper. 

Please carefully read the `train_badnet.py` and `configs.py`, then change the parameters for your experiment.  

## How to get teacher model?  
we obtained the teacher model by finetuning all layers of the backdoored model using 5% clean data with data augmentation techniques. In our paper, we only finetuning the backdoored model for 5~10 epochs. Please check more details of our experimental settings in section 4.1; The finetuning code is easy to get by just use the cls_loss to train it, which means the distillation loss to be zero in the training process.  



## Other source of backdoor attacks
#### Attack

**CL:** Clean-label backdoor attacks

- [Paper](https://people.csail.mit.edu/madry/lab/cleanlabel.pdf)
- [Can be modified from this pytorch implementation](https://github.com/MadryLab/cifar10_challenge)

**SIG:** A New Backdoor Attack in CNNS by Training Set Corruption Without Label Poisoning

- [Paper](https://ieeexplore.ieee.org/document/8802997/footnotes)


**Refool**: Reflection Backdoor: A Natural Backdoor Attack on Deep Neural Networks

- [Paper](https://arxiv.org/abs/2007.02343)
- [Code](https://github.com/DreamtaleCore/Refool)
- [Project](http://liuyunfei.xyz/Projs/Refool/index.html)

#### Defense

**MCR**: Bridging Mode Connectivity in Loss Landscapes and Adversarial Robustness

- [Paper](https://arxiv.org/abs/2005.00060)
- [Pytorch implementation](https://github.com/IBM/model-sanitization)

**Fine-tuning **: Defending Against Backdooring Attacks on Deep Neural Networks

- [Pytorch implementation1](https://github.com/VinAIResearch/input-aware-backdoor-attack-release/tree/master/defenses)

**Neural Attention Distillation **: Neural Attention Distillation: Erasing Backdoor Triggers from Deep Neural Networks 
- [Pytorch implementation1](https://github.com/bboylyg/NAD)

**Neural Cleanse**: Identifying and Mitigating Backdoor Attacks in Neural Networks

- [Paper](https://people.cs.uchicago.edu/~ravenben/publications/pdf/backdoor-sp19.pdf)
- [Tensorflow implementation](https://github.com/Abhishikta-codes/neural_cleanse)
- [Pytorch implementation1](https://github.com/lijiachun123/TrojAi)
- [Pytorch implementation2](https://github.com/VinAIResearch/input-aware-backdoor-attack-release/tree/master/defenses)

**STRIP**: A Defence Against Trojan Attacks on Deep Neural Networks

- [Paper](https://arxiv.org/pdf/1911.10312.pdf)
- [Pytorch implementation1](https://github.com/garrisongys/STRIP)
- [Pytorch implementation2](https://github.com/VinAIResearch/input-aware-backdoor-attack-release/tree/master/defenses)

#### Library

`Note`: TrojanZoo provides a universal pytorch platform to conduct security researches (especially backdoor attacks/defenses) of image classification in deep learning.

Backdoors 101 — is a PyTorch framework for state-of-the-art backdoor defenses and attacks on deep learning models. 

- [trojanzoo](https://github.com/ain-soph/trojanzoo)
- [backdoors101](https://github.com/ebagdasa/backdoors101)


## Contacts

If you have any questions, leave a message below with GitHub.

