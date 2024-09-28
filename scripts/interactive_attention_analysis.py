import sys
import os

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml
from tqdm import tqdm
from transformers import DebertaV2TokenizerFast, DebertaV2ForQuestionAnswering # change to AutoTokenizer, AutoModelForQuestionAnswering
from src.attention_analysis import qa_with_attended_token, visualize_attentions, visualize_single_attention_head
from src.utils import load_model_and_tokenizer

def main():
    
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
        
    experiments_path = os.path.abspath('models')
    experiments = [name for name in os.listdir(experiments_path) if os.path.isdir(os.path.join(experiments_path, name))]
    print("Available experiments:\n")
    for experiment_num in experiments:
        print(experiment_num)
    experiment = input("Enter the experiment name to visualize: ")
    if experiment not in experiments:
        print("Experiment not found.")
        return
        
    model, tokenizer = load_model_and_tokenizer(config['model_name'], f"models/{experiment}/best_model/") 
    
    while True:
        # Prompt the user for input
        question = input("Enter a question: ")
        response = input("Enter a context/response: ")
        vis_attns = input("Do you want to visualize the attentions? (yes/no): ").strip().lower()

        # Process the input with the model
        _, attentions = qa_with_attended_token(question, response, tokenizer, model)
        
        if vis_attns == 'yes':
            
            starting_layer = int(input("What layer do you want to start visualizing attentions from? (0-11): ").strip().lower())
            final_layer = int(input("What layer do you want to end visualizing attentions from? (0-11): ").strip().lower())
            # Error handling for starting_layer and final_layer
            
            # Validate that the layer numbers are within the valid range
            if not (0 <= starting_layer <= 11) or not (0 <= final_layer <= 11):
                raise ValueError("Layer numbers must be between 0 and 11.")

            if starting_layer > final_layer:
                raise ValueError("Starting layer must be less than or equal to final layer.")
            
            # Visualize the attentions
            for layer_num in tqdm(range(int(starting_layer), int(final_layer) + 1)):
                attention_matrix = attentions[layer_num]
                num_heads = attention_matrix.size(1)

                tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode_plus(question, response, add_special_tokens=True)['input_ids'])[:attention_matrix.size(-1)] # ensure correct size
                processed_tokens = [token.replace('▁', ' ').strip() for token in tokens]
                visualize_attentions(attention_matrix[0], processed_tokens, num_heads, layer_num, save_dir=f'models/{experiment}/attentions_vis/')
                
            print(f"Attentions visualized. See the 'models/{experiment}/attentions_vis' directory for the visualizations.")
            
            indiv_attn = input("Would you like to save an individual attention head? (yes/no): ").strip().lower()
            if indiv_attn == 'yes':
                
                while True: 
                    layer_num = int(input("Enter the layer number: "))
                    head_num = int(input("Enter the head number: "))
                    visualize_single_attention_head(attentions, processed_tokens, layer_num, head_num, save_dir=f'models/{experiment}/attentions_vis/individual_heads/')
                    response = input(f"Individual attention head saved at models/{experiment}/attentions_vis/individual_heads.\nWould you like to save another individual attention head? (yes/no): ")
                    if response != 'yes':
                        break
            else:
                print("Not visualizing individual attention heads.")       
        else:
            print("Not visualizing attentions.")
        # Ask if the user wants to input another question/context pair
        another = input("Do you want to analyze another question answer pair? (yes/no): ").strip().lower()
        if another != 'yes':
            break

if __name__ == "__main__":
    main()