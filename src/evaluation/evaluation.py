import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

def evaluate_and_infer(model, dataset, batch_size, device, seed_string, char_to_idx, idx_to_char, max_length, temperature=1.0, verbose=False):
    """
    Evaluate the model and generate text based on a seed string.

    Args:
        model (nn.Module): The trained language model.
        dataset (CharDataset): The evaluation dataset.
        batch_size (int): Batch size for evaluation.
        device (torch.device): Device (CPU or GPU) for evaluation and inference.
        seed_string (str): The initial text for text generation.
        char_to_idx (dict): Character-to-index mapping.
        idx_to_char (dict): Index-to-character mapping.
        max_length (int): Maximum length of generated text.
        temperature (float): Sampling temperature for text generation.

    Returns:
        dict: A dictionary containing evaluation loss, perplexity, and generated text.
    """
    # ====== Evaluation ======
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()

    model.to(device)
    model.eval()  # Set the model to evaluation mode

    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        print("Evaluating the model...")
        for input_seq, target_seq in tqdm(data_loader, bar_format="{bar:30} {percentage:3.0f}% "):
            input_seq, target_seq = input_seq.to(device), target_seq.to(device)

            # Forward pass
            logits = model(input_seq)  # Shape: (batch_size, seq_len, vocab_size)

            # Compute loss
            loss = criterion(logits.view(-1, logits.size(-1)), target_seq.view(-1))
            total_loss += loss.item() * target_seq.numel()  # Weighted by number of tokens
            total_tokens += target_seq.numel()

    # Compute average loss and perplexity
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss))

    if verbose:
        print(f"Evaluation Complete:\n- Loss: {avg_loss:.4f}\n- Perplexity: {perplexity:.4f}")

    # ====== Text Generation ======
    input_indices = [char_to_idx[char] for char in seed_string]
    input_tensor = torch.tensor(input_indices, dtype=torch.long, device=device).unsqueeze(0)

    generated_text = seed_string

    with torch.no_grad():
        print("\nGenerating text...")
        for _ in range(max_length):
            if input_tensor.size(1) > model.position_embedding.num_embeddings:
                input_tensor = input_tensor[:, -model.position_embedding.num_embeddings:]

            logits = model(input_tensor)
            logits = logits[:, -1, :]  # Focus on the last token
            logits = logits / temperature
            probabilities = torch.softmax(logits, dim=-1)
            next_token_idx = torch.multinomial(probabilities, num_samples=1).item()

            generated_text += idx_to_char[next_token_idx]
            input_tensor = torch.cat((input_tensor, torch.tensor([[next_token_idx]], device=device)), dim=1)
    if verbose:
        print("\nGenerated Text:")
        print(generated_text)

    return {"loss": avg_loss, "perplexity": perplexity.item(), "generated_text": generated_text}