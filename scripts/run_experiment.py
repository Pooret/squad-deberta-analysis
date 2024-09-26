import sys
import os

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AdamW, get_linear_schedule_with_warmup
import torch
import yaml
from src.data_loading import load_data, preprocess_data
from src.model import build_model
from src.train import train_model
from src.utils import save_model


def main():
    
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    print(config)
    
    # set the device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS backend on Apple Silicon")
    else:
        device = torch.device("cpu")
        print("MPS backend not available, using CPU instead")

    # load the data and tokenizer
    dataset, tokenizer = load_data()
    
    # preprocess the data
    processed_dataset = preprocess_data(dataset, tokenizer, max_length=config['max_length'], stride=config['stride'], padding=config['padding'])
    
    # build the model
    model = build_model(model_name=config['model_name']).to(device)
    
    # initialize the optimizer, scheduler, scaler 
    optimizer = AdamW(model.parameters(), lr=float(config['learning_rate']))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(processed_dataset['train']) * config['epochs'])
   
    # train the model
    train_model(model, processed_dataset, tokenizer, optimizer, scheduler, device, config)
    
if __name__ == "__main__":
    main()