from cs336_basics.transformer import Transformer
from cs336_basics.embedding import Embedding
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.linear import Linear
from cs336_basics.softmax import softmax
from jaxtyping import Float, Int
from torch import Tensor, nn

class TransformerLM(nn.Module):
    def __init__(self, 
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
    ) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
        """
        Runs a forward pass of a Transformer language model with the given parameters and weights.
        Args:
            vocab_size (int): The size of the vocabulary (number of unique tokens).
            context_length (int): The maximum sequence length that the model will be trained on.
            d_model (int): The dimensionality of the input and output features.
            num_layers (int): The number of Transformer layers in the model.
            num_heads (int): The number of attention heads in each Transformer layer.
            d_ff (int): The dimensionality of the feedforward network in each Transformer layer.
            rope_theta (float): The base frequency for the rotary positional embeddings.
        Returns:
            Float[Tensor, " batch_size sequence_length vocab_size"]: A tensor containing the logits for each token in the vocabulary. Shape: (batch_size, sequence_length, vocab_size).
        """
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model)
        self.transformer_layers = nn.ModuleList([
            Transformer(d_model, num_heads, d_ff, max_seq_len=context_length, theta=rope_theta) for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.output = Linear(d_model, vocab_size)

    def forward(self, x: Int[Tensor, " batch_size sequence_length"]) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
        """
        Computes the forward pass of the Transformer language model.
        Args:
            x (Int[Tensor, " batch_size sequence_length"]): A tensor containing the token IDs for the input sequences. Shape: (batch_size, sequence_length).
        Returns:
            Float[Tensor, " batch_size sequence_length vocab_size"]: A tensor containing the logits for each token in the vocabulary. Shape: (batch_size, sequence_length, vocab_size).
        """
        x = self.embedding(x)
        for layer in self.transformer_layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.output(x)
        return logits
