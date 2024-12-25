# smaLLM: A Character-Level Language Model

smaLLM is a Transformer-based language model designed to perform character-level language modeling. It can be trained on text datasets to learn patterns and generate coherent sequences of text based on a seed string. The model also evaluates its performance using cross-entropy loss and perplexity.

---

## **File Structure**

```plaintext
src/
│
├── models/
│   ├── transformer_block.py       # TransformerBlock definition
│   ├── transformer_model.py       # TransformerLanguageModel definition
|   | - Trained models -
│   ├── smaLLM_lr_decay_01.pth
│   ├── smaLLM_lr_decay_02.pth
│   ├── smaLLM_no_lr_decay_01.pth
│
├── training/
│   ├── training.py                   # Training function and script
│
├── evaluation/
│   ├── evaluation.py              # Evaluation and inference function
│
├── datasets/
│   ├── char_dataset.py            # CharDataset definition
│   ├── data.txt                   # Working Dataset
│
├── run_training.py                # Main script to start training
├── run_inference.py               # Script for evaluation and text generation
│
└── utils/
    ├── graph.py                   # Utility script to generate graphs (e.g., loss per epoch)
```

---

## **Requirements**

The project requires the following dependencies:

- Python 3.12
- PyTorch 2.4.1
- Argparse (for command-line arguments)
- TQDM (for progress bars)

Install the dependencies via pip:
```bash
pip install torch argparse tqdm
```

---

## **Usage**

### **1. Dataset Preparation**

Ensure that the dataset (`data.txt`) is placed in the `datasets/` folder. The dataset should be a plain text file that the `CharDataset` class can process.

---

### **2. Training the Model**

To train the model, run:
```bash
python run_training.py
```
This will:
- Load the dataset.
- Train the smaLLM model for a specified number of epochs.
- Save the trained model to `smaLLM_lr_decay_01.pth`.

---

### **3. Evaluating the Model**

To evaluate the model and compute perplexity, run:
```bash
python run_evaluation.py --model_path <path_to_model> --seed_string "<seed_text>"
```

**Example:**
```bash
python run_evaluation.py --model_path smaLLM_lr_decay_01.pth --seed_string "O God, O God!"
```

This will:
- Compute cross-entropy loss and perplexity on the dataset.
- Generate text based on the provided seed string.

---

### **4. Generating Text**

Use the `run_evaluation.py` script to generate text:
```bash
python run_evaluation.py --seed_string "<your_seed_text>" --max_length 200
```

- `--seed_string`: Initial string to seed the text generation.
- `--max_length`: Maximum number of characters to generate.
- `--temperature`: Adjusts randomness during generation (higher = more diverse output).

---

## **Implementation Details**

### **Model Architecture**
- **TransformerBlock**: Implements multi-head self-attention and feed-forward layers.

- **TransformerLanguageModel**: Composes multiple Transformer blocks with embeddings and a final output layer.

### **Training Details**
- Optimizer: Adam
- Learning Rate: Initial value set to `0.001`.
- Learning Rate Scheduler: Exponential decay with `gamma=0.95`.
- Gradient Clipping: Norm value set to `1.0`.

### **Evaluation Metrics**
- **Cross-Entropy Loss**: Measures prediction error on the test dataset.
- **Perplexity**: Quantifies how well the model predicts sequences $$PPL = e^{\text{Loss}}$$

### **Text Generation**
- Sampling can be controlled using the `temperature` parameter.
- Dynamically adjusts input sequences exceeding the model's `block_size`.

---

## **Visualization**

The `utils/graph.py` script provides tools for visualizing metrics like:
- Loss per epoch
- Effect of learning rate decay

Run the script as:
```bash
python utils/graph.py
```
---
*Dany A. Darghouth - December 2024*
