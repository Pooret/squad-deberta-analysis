import os
from transformers import DebertaV2ForQuestionAnswering, DebertaV2TokenizerFast # change to AutoModelForQuestionAnswering, AutoTokenizer
def save_model(model, output_dir):
    """
    Save the model to the provided output directory.
    
    Args:
        model (nn.Module): The model to save.
        output_dir (str): Directory to save the model.
    """
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")
    
def load_model_and_tokenizer(model_name, output_dir):
    """
    Load the model from the provided output directory.
    
    Args:
        model_name (str): The model name to load.
        output_dir (str): Directory to load the model from.
        
    Returns:
        nn.Module: The loaded model.
    """
    model = DebertaV2ForQuestionAnswering.from_pretrained(output_dir) # change to AutoModelForQuestionAnswering
    tokenizer = DebertaV2TokenizerFast.from_pretrained(model_name) # change to AutoTokenizer
    
    if model is not None:
        print(f"Model loaded from {output_dir}")
    return model, tokenizer