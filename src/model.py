from transformers import DebertaV2ForQuestionAnswering # change to AutoModelForQuestionAnswering

def build_model(model_name='microsoft/deberta-v3-base'):
    model = DebertaV2ForQuestionAnswering.from_pretrained(model_name)
    return model