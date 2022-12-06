import matplotlib.pyplot as plt
import json
import numpy as np
from scipy.ndimage.filters import uniform_filter1d

file_path = 'kinematic_model.txt'

loss = []

with open(file_path, 'r') as f:
    for line in f:       
        if len(line.split("Loss: ")) > 1:
            if line.split("Loss: ")[1].strip() != "nan":
                loss.append(float(line.split("Loss: ")[1].strip()))

x = np.arange(0, len(loss), 1)

plt.plot(x, uniform_filter1d(loss, size=1))
plt.title(f'Training Loss vs. Step')
plt.xlim(0, 250)
plt.ylim(0,1)
plt.show()