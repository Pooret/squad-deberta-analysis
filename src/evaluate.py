import torch


# Function to compute EM and F1 score during validation
def compute_metrics(start_pred, end_pred, start_true, end_true):
    """
    Compute Exact Match (EM) and F1 scores for predicted and true spans.

    Args:
        start_pred (list of int): Predicted start indices.
        end_pred (list of int): Predicted end indices.
        start_true (list of int): True start indices.
        end_true (list of int): True end indices.

    Returns:
        tuple: Exact Match (EM) score and F1 score.
    """
    em = 0
    f1 = 0
    n = len(start_pred)
    for i in range(n):
        pred_span = set(range(start_pred[i], end_pred[i] + 1))
        true_span = set(range(start_true[i], end_true[i] + 1))

        # Exact Match
        em += int(pred_span == true_span)

        # F1 Score
        overlap = pred_span & true_span
        if len(overlap) > 0:
            precision = len(overlap) / len(pred_span)
            recall = len(overlap) / len(true_span)
            f1 += 2 * (precision * recall) / (precision + recall)

    em = em / n
    f1 = f1 / n
    return em, f1

def validate_model(model, val_loader, device, compute_metrics):
    """
    Validate the model on the validation set.
    
    Args:
        model (nn.Module): The model to validate.
        val_loader (DataLoader): DataLoader for the validation set.
        device (torch.device): Device to use for validation (CPU or GPU).
        compute_metrics (function): Function to compute evaluation metrics.
    
    Returns:
        tuple: Average validation loss, exact match (EM) score, and F1 score.
    """
    model.eval()
    val_loss = 0.0
    em_score = 0.0
    f1_score = 0.0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            start_positions=start_positions,
                            end_positions=end_positions)

            loss = outputs.loss
            val_loss += loss.item()

            # Compute Metrics
            start_preds = outputs.start_logits.argmax(dim=-1).cpu().numpy()
            end_preds = outputs.end_logits.argmax(dim=-1).cpu().numpy()

            start_true = start_positions.cpu().numpy()
            end_true = end_positions.cpu().numpy()

            em, f1 = compute_metrics(start_preds, end_preds, start_true, end_true)
            em_score += em
            f1_score += f1

    avg_val_loss = val_loss / len(val_loader)
    avg_em = em_score / len(val_loader)
    avg_f1 = f1_score / len(val_loader)
    return avg_val_loss, avg_em, avg_f1