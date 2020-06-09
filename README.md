# Lipschitz-Certifiable Training with a Tight Outer Bound (BCP)

This repository is the official implementation of BCP, Box Constraints Propagation (Lipschitz-Certifiable Training with a Tight Outer Bound).

<!----
> 📋Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials
---->

## Requirements

To install requirements:

```setup
conda env create -f environment.yml
```

<!----
> 📋Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...
---->

## Training (and Evaluation)

To train and evaluate the model(s) in the paper, run this command:

```train
python main_mnist.py
python main_cifar10.py
```

<!----
> 📋Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.
---->


## Pre-trained Models

You can download pretrained models here:

- [BCP model](https://drive.google.com/file/d/17MsumEnGQvpMQaXMXRZK4xK8mpnO0oRz/view?usp=sharing) trained on MNIST.
- [BCP model](https://drive.google.com/file/d/1MuXNJ63_HwzKtBMrRlvrLGIzD3FhH-Ov/view?usp=sharing) trained on CIFAR-10.


<!----
> 📋Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.
---->

## Evaluation of pretrained models

After downloading the pretrained models to the directory ./pretrained, you are ready to evaluate them.
To evaluate the pretrained model, run:

```eval
python evaluation_mnist.py --test_pth pretrained/mnist_save.pth
python evaluation_cifar10.py --test_pth pretrained/cifar10_save.pth
```

<!----
> 📋Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).
---->

## Results

Our model achieves the following performance against $\ell_2$-perturbation :

### MNIST ($\epsilon_2$=1.58)

| Model name         | Standard  | PGD^100 | Verification  |
| ------------------ |---------------- | -------------- | --------------  |
| BCP                |     92.54%         |      66.23%       | 48.20%  |
| [CAP](https://arxiv.org/abs/1805.12514)                |     88.39%         |      62.25%       | 43.95%  |
| [LMT](https://arxiv.org/abs/1802.04034)               |     86.48%         |      53.56%       | 40.55%  |

### CIFAR-10 ($\epsilon_2$=36/255)

Model1

| Model name         | Standard  | PGD^100 | Verification  |
| ------------------ |---------------- | -------------- | --------------  |
| BCP                |     65.64         |      59.59%       | 50.27%  |
| [CAP](https://arxiv.org/abs/1805.12514)                |     60.14%         |      55.67%       | 50.29%  |
| [LMT](https://arxiv.org/abs/1802.04034)               |     56.49%         |      49.83%       | 37.20%  |

Model2

| Model name         | Standard  | PGD^100 | Verification  |
| ------------------ |---------------- | -------------- | --------------  |
| BCP                |     65.72%         |      60.78%       | 51.30%  |
| [CAP](https://arxiv.org/abs/1805.12514)                |     60.10%         |      56.20%       | 50.87%  |
| [LMT](https://arxiv.org/abs/1802.04034)               |     63.05%         |      58.32%       | 38.11%  |

### Tiny ImageNet ($\epsilon_2$=36/255)

| Model name         | Standard  | PGD^100 | Verification  |
| ------------------ |---------------- | -------------- | --------------  |
| BCP                |     28.76%         |      26.64%       | 20.08%  |

<!----
> 📋Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 
---->

<!----
## Contributing
> 📋Pick a licence and describe how to contribute to your code repository. 
---->
