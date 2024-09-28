import os
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def qa_with_attended_token(question, response_with_context, tokenizer, model):
    inputs = tokenizer.encode_plus(question, response_with_context, return_tensors='pt', add_special_tokens=True)
    outputs = model(**inputs, output_attentions=True)
    attentions = outputs.attentions

    start_logits = outputs.start_logits
    end_logits = outputs.end_logits

    start_probs = torch.softmax(start_logits, dim=-1)
    end_probs = torch.softmax(end_logits, dim=-1)

    start_index = torch.argmax(start_probs)
    end_index = torch.argmax(end_probs)

    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

    answer_tokens = tokens[start_index: end_index + 1]
    answer = " ".join(answer_tokens)

    num_layers = len(attentions)
    num_heads = attentions[0].size(1)

    question_end_index = tokens.index('[SEP]')
    summed_attentions = np.zeros(len(tokens[:question_end_index]))

    for layer in range(num_layers):
        for head in range(num_heads):
            summed_attentions += attentions[layer][0, head, start_index:end_index + 1, :question_end_index].sum(dim=0).detach().numpy()
        summed_attentions /= np.sum(summed_attentions)

    special_tokens = {'[CLS]', '[SEP]', '[PAD]', '[UNK]'}
    token_attention_pairs = [(token, summed_attentions[i]) for i, token in enumerate(tokens[:question_end_index]) if token not in special_tokens]

    token_attention_pairs.sort(key=lambda x: x[1], reverse=True)

    answer = answer.replace(' ▁', ' ').replace('▁', '  ').strip()
    attended_token = token_attention_pairs[0][0].replace(' ▁', ' ').replace('▁', '  ').strip()

    # tokens_and_attns = [val for val in [(token, summed_attentions[i]) for i, token in enumerate(tokens[:question_end_index])]]

    print(f"Question: {question}")
    print(f"Answer: {answer}")
    print(f"Main attended token: {attended_token}")

    return token_attention_pairs, attentions

def visualize_attentions(attention, tokens, num_heads, layer_num, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    plt.figure(figsize=(20, 16))
    for head in range(num_heads):
        attention_head = attention[head].detach().numpy()
        ax = plt.subplot(4, 3, head + 1)
        sns.heatmap(attention_head, xticklabels=tokens, yticklabels=tokens, ax=ax)
        ax.set_title(f'Layer {layer_num} - Head {head}')
    
    plt.tight_layout()
    
    file_path = os.path.join(save_dir, f'layer_{layer_num}.png')
    plt.savefig(file_path)
    plt.close()  # Close the plot to free up memory
    
def visualize_single_attention_head(attentions, tokens, layer_num, head_num, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    attention_head = attentions[layer_num][0][head_num].detach().numpy()

    plt.figure(figsize=(28, 28))
    ax = plt.axes()
    sns.heatmap(attention_head, xticklabels=tokens, yticklabels=tokens, ax=ax, cbar=False)
    ax.set_title(f'Layer {layer_num} - Head {head_num}', fontsize=32)

    ax.tick_params(axis='x', labelsize=28)
    ax.tick_params(axis='y', labelsize=28)
    
    file_path = os.path.join(save_dir, f'layer_{layer_num}_head_{head_num}.png')
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()  # Close the plot to free up memory

