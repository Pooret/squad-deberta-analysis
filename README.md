# Question Answering with DeBERTa and Attention Visualization

This project implements a question-answering system using Microsoft's DeBERTa model. It includes functionalities to train the model on the SQuAD dataset, analyze the attention mechanisms within the transformer layers, and visualize these attentions to gain insights into the model's focus during inference.



The way in which a model processes and understands langauge is crucial for understanding how these models work. In this project, an encoder transformer was pretrained to answer questions given some context. The folowing is a sample of the trained model's output:

*Question:* What is my favorite color?  
*Answer:* My favorite color is red.  

*Question:* What is my favorite color?  
*Model Answer*: red  
*Main Attended Token in Question:* What  


Below is a heatmap that visualizes the attention heads in layer 7. Note that Heads 4, 5, and 11 have attention from "What" in the y-axis to the "is" and "red" on the x-axis.  


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


## Configuration

    model_name: 'microsoft/deberta-v3-base'
    epochs: 3
    learning_rate: 3e-5
    batch_size: 16
    max_length: 512
    stride: 128
    padding: 'longest'
    patience: 2
    best_model_path: 'models/experiment_1/best_model'

- model_name: Pretrained model to use.  
- epochs: Number of training epochs.  
- learning_rate: Learning rate for the optimizer.  
- batch_size: Batch size for training and validation.  
- max_length: Maximum sequence length for tokenization.  
- stride: Stride size for handling long sequences.  
- padding: Padding strategy ('longest', 'max_length', etc.).  
- patience: Patience for early stopping.  
- best_model_path: Directory to save the best model checkpoint.  


## Results

### Attention Visualization
The attention visualizations help in understanding which parts of the input the model focuses on when answering a question. For example, in Layer 7, we observe that the model attends heavily to certain keywords in the context that are crucial for generating the correct answer.

In this heatmap, darker colors represent higher attention weights. The model is focusing on tokens like "transformer", "layers", and "attention", indicating their importance in generating the answer.

### Performance Metrics
After training the model for 10, the model achieved the following performance on the validation set:

Validation Loss: 0.788
Exact Match (EM): 64.8%
F1 Score: 83.3%
These metrics indicate the model's ability to predict the correct answer spans.


## Acklowedgements
 - Preston Brown, without whom this project wouldn't be possible
 - Stanford Question and Answering Dataset
 - HuggingFace Transformers Library