import torch
from generation.generation import generate
from models.smaLLM import load_model
from datasets.charDataset import charDataset
import argparse

# ====== Argument Parser ======
parser = argparse.ArgumentParser(description="Evaluate and Generate Text using a pre-trained smaLLM model.")
parser.add_argument("--data_path", type=str, default="datasets/data_elon.txt", help="Path to the dataset.")
parser.add_argument("--seed_string", type=str, default="I think", help="Seed string to start generating text.")
parser.add_argument("--max_length", type=int, default=280, help="Maximum length of the generated text.")
parser.add_argument("--temperature", type=float, default=1.0, help="Temperature value for sampling. Higher values increase diversity.")

# ====== Configuration ======
model_path = parser.parse_args().model_path
data_path = parser.parse_args().data_path
seed_string = parser.parse_args().seed_string
max_length = parser.parse_args().max_length
temperature = parser.parse_args().temperature

# Model hyperparameters (has to be same as training, current correspond to pre-trained model)
embed_dim = 128
num_heads = 4
ff_dim = 256
num_blocks = 4
block_size = 128
dropout = 0.1

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\x1B[3mUsing device: {device}\x1B[0m")

# ====== Load Dataset ======
with open(data_path, "r") as f:
    data = f.read()

dataset = charDataset(data = data, config = None, block_size = block_size)

# ====== Load Model ======
model = load_model(
    model_path=model_path,
    vocab_size=dataset.get_vocab_size(),
    embed_dim=embed_dim,
    num_heads=num_heads,
    ff_dim=ff_dim,
    num_blocks=num_blocks,
    block_size=block_size,
    dropout=dropout,
    device=device
)

# ====== Generate Text ======
print("\33[1mGenerating Text...\33[0m")
results = generate(
    model=model,
    device=device,
    seed_string=seed_string,
    char_to_idx=dataset.stoi,
    idx_to_char=dataset.itos,
    max_length=max_length,
    temperature=temperature
)

# ====== Display Results ======
print("\n\33[1mFinal Results:\33[0m")
print(f"\33[1m- Generated Text:\33[0m\n{results['generated_text']}")

print('_____\n\x1B[3m Dany A. Darghouth - December 2024')