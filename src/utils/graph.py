import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import datetime

sns.set(font='Serif')

lr_decay_01_loss = [1.4486, 1.3294, 1.3053, 1.2908, 1.2803]
# lr_decay_01_LR = [0.000950, 0.000902, 0.000857, 0.000814, 0.000773]

# If the model finishes training early enough, use this line
#lr_decay_01_12_blocks_768_dim_loss = [0, 0, 0.0, 0.0, 0.0]

lr_decay_02_loss = [1.4536, 1.3392, 1.3152, 1.3020, 1.2929]
# lr_decay_02_LR = [0.000990, 0.000980, 0.000970, 0.000961, 0.000951]

lr_no_decay_loss = [1.4536, 1.3301, 1.3091, 1.2971, 1.2891]

epochs = np.arange(1, 6, 1)

# Plot the loss
plt.figure(figsize=(10, 6))

plt.xticks(np.arange(1, 6, 1))

plt.plot(epochs, lr_decay_01_loss, label="LR Decay = 0.95 , 4 blocks, 128 embedded dimensions", marker='o', color='darkred')
plt.plot(epochs, lr_decay_02_loss, label="LR Decay = 0.99, 4 blocks, 128 embedded dimensions", marker='o',color='darkblue')
plt.plot(epochs, lr_no_decay_loss, label="No LR Decay, 4 blocks, 128 embedded dimensions", marker='o', color='darkgreen')
#plt.plot(epochs, lr_decay_01_12_blocks_768_dim_loss, label="LR Decay = 0.95, 12 blocks, 768 embedded dimensions", marker='o', color='darkorange')

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.suptitle("Training Loss, Learning Rate Decay Comparison", fontsize=18, fontweight='bold')
plt.title("smaLLM on a character-level dataset, with 0.0001 starting learning rate", fontsize=12)
plt.legend()
plt.grid(True)
date_format_string = "%Y-%m-%d_%H-%M-%S"
plt.savefig(f'loss_lr_decay_comparison_{datetime.datetime.now().strftime(date_format_string)}.png')
plt.show()


