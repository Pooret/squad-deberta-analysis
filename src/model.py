from transformers import DebertaV2ForQuestionAnswering

def build_model(model_name='microsoft/deberta-v3-base'):
    model = DebertaV2ForQuestionAnswering.from_pretrained(model_name)
    return model