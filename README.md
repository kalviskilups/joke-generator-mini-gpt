## Reddit Joke Generator using Simplified Transformer-Based Language Model

Welcome to the repository for the "Reddit Joke Generator," implemented using PyTorch and a simplified transformer architecture! This repository aims to replicate the 2017 paper "Attention Is All You Need" by Ashish Vaswani et al., presenting a simplified version akin to NanoGPT.

## Code
The cleaning_script.py file contains code to parse JSON data and write it into a .txt file. Additionally, the hyperparameters.py file comprises defined hyperparameters, data cleaning functions, as well as encoding and decoding functions. main.py contains the model architecture and the driver code.

## Results
After 5000 iterations, the achieved results were not particularly impressive, and the loss did not converge to a minimum. However, after 10000 iterations, the outcomes showed improvement with fewer grammatical errors. Nevertheless, the generated jokes were not easily understandable or amusing to the author. You can also experiment by adjusting parameters, such as modifying the context length used for predictions. The author attempted to set the context length to 512 tokens, which might yield better predictions, but generating jokes might not be the ideal use case. Providing multiple books that are similar as input might offer improved results.
