# xy_hpp
This repository is the source of the hierarchical diffusion-based planner. Our framework was inspired by the "Diffusion-Based Planning for Autonomous Driving with Flexible Guidance". [link](https://github.com/ZhengYinan-AIR/Diffusion-Planner.git) And the framework is developed based on the Diffusion planner. We realeased the summary code as a record of our framework, which will be organized well in the future.

# Environment Setup
1. Setup the nuplan dataset following the official guidance.
2. Setup the environment on Ubuntu 20.04

```
conda create -n planner python=3.9
conda activate planner

# install nuplan-devkit
git clone https://github.com/motional/nuplan-devkit.git && cd nuplan-devkit
pip install -e .
pip install -r requirements.txt

cd ..
git clone https://github.com/HITXCI/xy_hpp.git
pip install -e .
pip install -r requirements_torch.txt
```
