import torch
import json
from torch.utils.data import random_split


class HyperparametersAndDataEncoding:
    """
    Handles hyperparameters and data encoding for the Toy GPT Language Model.
    """
    def __init__(self, file_path, data_path):
        """
        Initializes HyperparametersAndDataEncoding object.

        Args:
        - file_path (str): Path to the JSON file containing hyperparameters.
        - data_path (str): Path to the file containing text data.
        """

        with open(file_path, 'r', encoding = "utf-8") as file:
            hyperparameters = json.load(file)
        
        with open(data_path, 'r', encoding = "utf-8") as file:
            text = file.read()

        # Hyperparameters
        self.batch_size = hyperparameters['batch_size']
        self.block_size = hyperparameters['block_size']
        self.max_iterations = hyperparameters['max_iterations']
        self.evaluation_interval = hyperparameters['evaluation_interval']
        self.learning_rate = hyperparameters['learning_rate']
        self.device = hyperparameters['device']
        self.evaluation_iterations = hyperparameters['evaluation_iterations']
        self.embedding_dimension = hyperparameters['embedding_dimension']
        self.num_heads = hyperparameters['num_heads']
        self.num_layers = hyperparameters['num_layers']
        self.dropout_rate = hyperparameters['dropout_rate']

        # Data
        self.characters = sorted(list(set(text)))
        self.vocab_size = len(self.characters)
        self.stoi = {ch: i for i, ch in enumerate(self.characters)}
        self.itos = {i: ch for i, ch in enumerate(self.characters)}
        self.data = torch.tensor(self.encode_text(text), dtype = torch.long)
        self.train_data, self.val_data = random_split(
        self.data,
        [int(0.9 * len(self.data)), (len(self.data) - int(0.9 * len(self.data)))]
        )

    def encode_text(self, s):
        """
        Encodes text to a list of indices.

        Args:
        - s (str): Input text.

        Returns:
        - List[int]: Encoded text as a list of indices.
        """

        return [self.stoi[c] for c in s]
    
    def decode_list(self, l):
        """
        Decodes a list of indices back to text.

        Args:
        - l (List[int]): List of indices.

        Returns:
        - str: Decoded text.
        """

        return ''.join([self.itos[i] for i in l])

    def get_batch(self, split):
        """
        Decodes a list of indices back to text.

        Args:
        - l (List[int]): List of indices.

        Returns:
        - str: Decoded text.
        """

        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i: i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1:i + self.block_size + 1] for i in ix])
        x, y = x.to(self.device), y.to(self.device)
        return x, y
