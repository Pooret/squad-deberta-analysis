# Question Answering with DeBERTa and Attention Visualization

This project implements a question-answering system using Microsoft's DeBERTa model. It includes functionalities to train the model on the SQuAD dataset, analyze the attention mechanisms within the transformer layers, and visualize these attentions to gain insights into the model's focus during inference.

![Visualization of attention weights in Layer 11](models/experiment_1/attentions_vis/layer_7.png)

## Table of Contents

- Features
- Installation
- Usage
    - Training the Model
    - Analyzing Attention Layers
- Project Structure
- Configuration
- Results
    - Attention Visualization
    - Performance Metrics
- Contributing
- License
- Acknowledgements 


## Features
- **Data Loading and Preprocessing**: Efficiently load and preprocess the SQuAD dataset for training.
- **Model Training**: Train DeBERTa for question-answering tasks with support for early stopping and model checkpointing.
- **Attention Analysis**: Extract and analyze attention weights from the transformer layers.
- **Visualization**: Visualize attention heatmaps to understand which tokens (words) the model focuses on when generating answers to questions.
- **Interactive Interface**: Input custom questions and answers to see the model's attentions for each layer.

## Installation
1. Clone the repository 
`git clone https://github.com/pooret/yourproject.git
cd yourproject
