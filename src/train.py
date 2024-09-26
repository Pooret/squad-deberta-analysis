import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from transformers import DataCollatorWithPadding
from .evaluate import validate_model, compute_metrics

def train_one_epoch(model, train_loader, optimizer, scheduler, device, epoch, config):
    """
    Train the model for one epoch.
    
    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for the training set.
        optimizer (Optimizer): Optimizer for the model.
        scheduler (Scheduler): Learning rate scheduler.
        device (torch.device): Device to use for training (CPU or GPU).
        epoch (int): The current epoch number.
        config (dict): Configuration dictionary from W&B.
    
    Returns:
        float: The average training loss for this epoch.
    """
    # Put model in training mode
    model.train()
    
    # keep track of loss
    train_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['epochs']} Training", leave=True)
    
    for batch_idx, batch in enumerate(progress_bar):
        
        # Load batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)
        
        # Zero grad the optimizer
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        start_positions=start_positions,
                        end_positions=end_positions)
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # update loss
        train_loss += loss.item()
        progress_bar.set_postfix(avg_loss=train_loss / (batch_idx + 1))
        
        # Log training updatges to W&B
        if (batch_idx + 1) % 100 == 0:
            wandb.log({"train_loss_step": train_loss / (batch_idx + 1)})
            
    avg_train_loss = train_loss / len(train_loader)
        
    return avg_train_loss
    
def train_model(model, processed_dataset, tokenizer, optimizer, scheduler, device, config):

    # initialize W&B
    wandb.init(project="question-answering", config=config)

    # for early stopping
    patience_counter = 0
    best_val_loss = float('inf')
    
    # initializd dataloaders
    train_loader = DataLoader(processed_dataset['train'], batch_size=config['batch_size'], shuffle=True, collate_fn=DataCollatorWithPadding(tokenizer, return_tensors='pt'))
    val_loader = DataLoader(processed_dataset['validation'], batch_size=config['batch_size'], shuffle=False, collate_fn=DataCollatorWithPadding(tokenizer, return_tensors='pt'))
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    
    # bewgin training loop
    for epoch in range(config['epochs']):
        
        # Train the model for one epoch
        avg_train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, epoch, config)
        
        # evaluate the model on the validation set
        avg_val_loss, avg_em, avg_f1 = validate_model(model, val_loader, device, compute_metrics)
        
        # log the metrics to W&B
        wandb.log({"epoch" : epoch + 1,
                     "train_loss": avg_train_loss,
                     "val_loss": avg_val_loss,
                     "val_em": avg_em,
                     "val_f1": avg_f1})
        
        print(f"Validation Loss: {avg_val_loss:.4f}, EM: {avg_em:.4f}, F1: {avg_f1:.4f}")
        
        # Early stopping and model checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"Saving new best model at epoch {epoch + 1}")
            epoch_model_path = os.path.join(config['best_model_path'], f'epoch_{epoch + 1}')
            if not os.path.exists(epoch_model_path):
                os.makedirs(epoch_model_path)
            model.save_pretrained(epoch_model_path)
            model.save_pretrained(config['best_model_path'])
            patience_counter = 0
            wandb.save(os.path.join(config['best_model_path'], '*'))
        else:
            patience_counter += 1
            
        if patience_counter >= config['patience']:
            print(f"Stopping early at epoch {epoch + 1}")
            break
        
        print(f"Epoch {epoch + 1}/{config['epochs']}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

    wandb.finish()