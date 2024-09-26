from transformers import DebertaV2TokenizerFast # change to AutoTokenizer
from datasets import load_dataset, DatasetDict

def load_data(train_size = None, test_size = None, dataset_name='squad', tokenizer_name='microsoft/deberta-v3-base'):
    """
    Load the specified dataset and tokenizer.

    Args:
        dataset_name (str): The name of the dataset to load.
        tokenizer_name (str): The name of the tokenizer to load.

    Returns:
        tuple: A tuple containing the dataset and the tokenizer.
    """
    tokenizer = DebertaV2TokenizerFast.from_pretrained(tokenizer_name)
    dataset = load_dataset(dataset_name)
    if train_size is not None: # fix this for error handling
        small_train_dataset = dataset['train'].select(range(train_size))
        small_val_dataset = dataset['validation'].select(range(test_size))
        small_dataset = {
        'train': small_train_dataset,
        'validation': small_val_dataset
            }
        small_dataset =  DatasetDict(small_dataset)
        return small_dataset, tokenizer
    else:
        return dataset, tokenizer

def preprocess_data(dataset, tokenizer, max_length=512, stride=128, padding="longest"):
    """
    Preprocess the dataset by tokenizing and aligning the answer spans with token indices.

    Args:
        dataset (Dataset): The dataset to preprocess.
        tokenizer (PreTrainedTokenizer): The tokenizer used for preprocessing.
        max_length (int): Maximum length of the tokenized inputs.
        stride (int): Stride size for handling long sequences.
        padding (str): Padding strategy to use ('longest', 'max_length', etc.).

    Returns:
        Dataset: The tokenized dataset with added start and end positions for the answers.
    """
    def preprocess_function(examples):
        tokenized_inputs = tokenizer(
            examples["question"],
            examples["context"],
            truncation="only_second",
            max_length=max_length,
            stride=stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=padding,
        )
        sample_mapping = tokenized_inputs.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized_inputs.pop("offset_mapping")

        tokenized_inputs["start_positions"] = []
        tokenized_inputs["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized_inputs["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)
            sequence_ids = tokenized_inputs.sequence_ids(i)

            sample_index = sample_mapping[i]
            answer = examples["answers"][sample_index]

            if len(answer["answer_start"]) == 0:
                tokenized_inputs["start_positions"].append(cls_index)
                tokenized_inputs["end_positions"].append(cls_index)
            else:
                start_char = answer["answer_start"][0]
                end_char = start_char + len(answer["text"][0])

                # Find the start token index
                token_start_index = 0
                while token_start_index < len(sequence_ids) and sequence_ids[token_start_index] != 1:
                    token_start_index += 1

                # Find the end token index
                token_end_index = len(input_ids) - 1
                while token_end_index >= 0 and sequence_ids[token_end_index] != 1:
                    token_end_index -= 1

                # If the answer is out of the span
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_inputs["start_positions"].append(cls_index)
                    tokenized_inputs["end_positions"].append(cls_index)
                else:
                    # Move the token_start_index and token_end_index to the actual start and end
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_inputs["start_positions"].append(token_start_index - 1)

                    while token_end_index > 0 and offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_inputs["end_positions"].append(token_end_index + 1)

        return tokenized_inputs

    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    return tokenized_dataset
