import torch
from evaluation.evaluation import evaluate_and_infer
from models.smaLLM import smaLLM, load_model
from datasets.charDataset import charDataset
import argparse

# ====== Argument Parser ======
parser = argparse.ArgumentParser(description="Evaluate and Generate Text using a pre-trained smaLLM model.")
parser.add_argument("--model_path", type=str, default="smaLLM_lr_decay_01.pth", help="Path to the pre-trained model.")
parser.add_argument("--data_path", type=str, default="datasets/data.txt", help="Path to the dataset.")
parser.add_argument("--seed_string", type=str, default="O God, O God!", help="Seed string to start generating text.")
parser.add_argument("--max_length", type=int, default=200, help="Maximum length of the generated text.")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation.")
parser.add_argument("--temperature", type=float, default=1.0, help="Temperature value for sampling. Higher values increase diversity.")

# ====== Configuration ======
model_path = parser.parse_args().model_path
data_path = parser.parse_args().data_path
seed_string = parser.parse_args().seed_string
max_length = parser.parse_args().max_length
batch_size = parser.parse_args().batch_size
temperature = parser.parse_args().temperature

# Model hyperparameters
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

# ====== Evaluate and Generate Text ======
print("\33[1mEvaluating and Generating Text...\33[0m")
results = evaluate_and_infer(
    model=model,
    dataset=dataset,
    batch_size=batch_size,
    device=device,
    seed_string=seed_string,
    char_to_idx=dataset.stoi,
    idx_to_char=dataset.itos,
    max_length=max_length,
    temperature=temperature
)

# ====== Display Results ======
print("\n\33[1mFinal Results:\33[0m")
print(f"\33[1m- Loss:\33[0m {results['loss']:.4f}")
print(f"\33[1m- Perplexity:\33[0m {results['perplexity']:.4f}")
print(f"\33[1m- Generated Text:\33[0m\n{results['generated_text']}")

print('_____\n\x1B[3m Dany A. Darghouth - December 2024')