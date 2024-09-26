# Question Answering with DeBERTa and Attention Visualization

This project implements a question-answering system using Microsoft's DeBERTa model. It includes functionalities to train the model on the SQuAD dataset, analyze the attention mechanisms within the transformer layers, and visualize these attentions to gain insights into the model's focus during inference.

![Visualization of attention weights in Layer 11](models/experiment_1/attentions_vis/layer_7.png)

## Table of Contents

- Features
- Usage
    - Training the Model
    - Analyzing Attention Layers
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

## Usage

1. Optionally edit the `config.yaml` file to set desired parameters if you wish to train the model. 
2. Run the training script `python run_experiment.py`. The program will ask you for an experiment name to run.
3. After training the model or loading in an old experiment, you can analyze and visualize the attention weights by running `python interactive_attention_analysis.py`
4. Enter a question and answer to that question with context.
5. Choose whether to visualize the attentions and which layers to view.
6. The model will the display the answer to the question and the word it pays the most attention to when providing the answer.


