import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

def generate(model, device, seed_string, char_to_idx, idx_to_char, max_length, temperature=1.0, verbose=False):
    """
    Evaluate the model and generate text based on a seed string.

    Args:
        model (nn.Module): The trained language model.
        device (torch.device): Device (CPU or GPU) for evaluation and inference.
        seed_string (str): The initial text for text generation.
        char_to_idx (dict): Character-to-index mapping.
        idx_to_char (dict): Index-to-character mapping.
        max_length (int): Maximum length of generated text.
        temperature (float): Sampling temperature for text generation.

    Returns:
        dict: generated text.
    """
    model.to(device)
    model.eval()  # Set the model to evaluation mode

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

    return {"generated_text": generated_text}