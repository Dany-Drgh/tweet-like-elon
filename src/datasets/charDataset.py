from torch.utils.data import Dataset
import torch

class charDataset(Dataset):
    """
    Emits batches of characters.
    Adapted from "https://github.com/karpathy/minGPT".
    """

    def __init__(self, config, data, block_size=128):
        """
        Args:
            config: dict
                Configuration object

            data: str
                Input data

            block_size: Int
                Size of the sequence length
        """

        chars = sorted(list(set(data))) # get characters from the input data
        self.stoi = {ch: i for i, ch in enumerate(chars)}  # Character to index
        self.itos = {i: ch for i, ch in enumerate(chars)}  # Index to character (inverse mapping, useful for decoding)
        self.block_size = block_size
        self.data = data
        self.vocab_size = len(chars)
        self.encoded_data = torch.tensor([self.stoi[ch] for ch in data], dtype=torch.long)

    def get_vocab_size(self):
        return self.vocab_size

    def __len__(self):
        # Number of chunks of size `block_size + 1` in the dataset
        return len(self.encoded_data) - self.block_size

    def __getitem__(self, idx):
        """
        Grabs a chunk of length `block_size + 1` starting from `idx`, encodes the
        characters to integers, and returns the input and target sequences as tensors.

        Args:
            idx: Int
                Index of the chunk

        Returns:
            (input_seq, target_seq): Tuple of torch.Tensor
                Tuple containing the input sequence and the shifted target sequence
        """
        
        chunk = self.encoded_data[idx : idx + self.block_size + 1]
        
        # Split into input (all but last) and target (all but first)
        input_seq = chunk[:-1]
        target_seq = chunk[1:]

        return input_seq, target_seq


# Test the CharDataset class
if __name__ == "__main__":
    # Test data: a small example string
    test_data = "To be or not to be, that is the question."

    # Instantiate the dataset
    block_size = 10  # Length of each input sequence
    dataset = charDataset(data=test_data, config = None, block_size=block_size)

    # Print vocabulary size
    if dataset.get_vocab_size() == len(set(test_data)):
        print(f"\33[92m\33[1mVocabulary size: {dataset.get_vocab_size()}\33[0m")
    else:
        print(f"\33[91m\33[1mVocabulary size: {dataset.get_vocab_size()}\33[0m")

    # Print total number of samples
    if len(dataset) == len(test_data) - block_size:
        print(f"\33[92m\33[1mTotal samples: {len(dataset)}\33[0m")
    else:
        print(f"Total samples: {len(dataset)}\n")

    # Print a few samples from the dataset
    for idx in range(3):  # Print first 3 samples
        input_seq, target_seq = dataset[idx]
        print(f"Sample {idx}:")
        print(f"Input: {''.join([dataset.itos[i.item()] for i in input_seq])}")
        print(f"Target: {''.join([dataset.itos[i.item()] for i in target_seq])}\n")