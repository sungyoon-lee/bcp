# Lipschitz-Certifiable Training with a Tight Outer Bound

This repository is the official implementation of Lipschitz-Certifiable Training with a Tight Outer Bound.

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

## Evaluation of pretrained models

To evaluate the pretrained model, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

<!----
> 📋Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).
---->

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

<!----
> 📋Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.
---->

## Results

Our model achieves the following performance on :

### MNIST

| Model name         | Standard Accuracy  | PGD Accuracy | Verification Accuracy  |
| ------------------ |---------------- | -------------- | --------------  |
| BCP                |     92.41%         |      64.70%       | 47.95%  |
| [Wong et al.](https://arxiv.org/abs/1805.12514)                |     88.39%         |      62.25%       | 43.95%  |
| BCP                |     86.48%         |      53.56%       | 40.55%  |

<!----
> 📋Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 
---->

<!----
## Contributing
> 📋Pick a licence and describe how to contribute to your code repository. 
---->
