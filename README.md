# How Physics and Background Attributes Impact Video Transformers in Robotic Manipulation: A Case Study on Pushing (IROS 2024)

<a href='https://arxiv.org/pdf/2310.02044'><img src='https://img.shields.io/badge/ArXiv-2303.09535-red'></a> 
<a href='https://cloudgripper.org/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> 

## Abstract
As model and dataset sizes continue to scale in robot learning, the need to understand what is the specific
factor in the dataset that affects model performance becomes increasingly urgent to ensure cost-effective data collection and model performance. In this work, we empirically investigate how physics attributes (color, friction coefficient, shape) and scene background characteristics, such as the complexity and dynamics of interactions with background objects, influence
the performance of Video Transformers in predicting planar pushing trajectories. We aim to investigate three primary
questions: How do physics attributes and background scene characteristics influence model performance? What kind of changes in attributes are most detrimental to model generalization? What proportion of fine-tuning data is required to adapt
models to novel scenarios? To facilitate this research, we present CloudGripper-Push-1K, a large real-world vision-based robot pushing dataset comprising 1278 hours and 460,000 videos of planar pushing interactions with objects with different physics and background attributes. We also propose Video Occlusion Transformer (VOT), a generic modular video-transformerbased trajectory prediction framework which features 3 choices of 2D-spatial encoders as the subject of our case study.

<img src="push1k.gif"  width="600" height="400">

## Dataset
[Example Dataset](https://cloudgripper.org/)

[Full 1.4T dataset](replace)

## Installation
```bash
conda create -n vot python=3.9
conda activate vot
pip install -r requirements.txt
```

## Training on SLURM
```bash
sbatch example.sh
```