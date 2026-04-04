from cs336_basics.top_p_sampling import top_p_sampling
import torch


@torch.no_grad()
def decoder(
    model: torch.nn.Module, 
    context_length: int,
    prompt: list[int],
    eos_token_id: int,
    p: float, 
    temperature: float = 1.0,
    max_tokens: int = 50
) -> torch.Tensor:
    """
    Generate a completion from a tokenized prompt using autoregressive top-p sampling.

    Args:
        model: The language model used to predict the next token at each decoding step.
        context_length: The maximum number of recent tokens to pass to the model.
        prompt: A list of token IDs representing the initial prompt.
        eos_token_id: The token ID that marks the end of generation.
        p: The cumulative probability threshold for top-p sampling. Must be in the range (0, 1].
        temperature: The temperature for scaling the logits. Must be greater than 0.
        max_tokens: The maximum number of new tokens to generate.

    Returns:
        A tensor of shape (1, prompt_length + generated_length) containing the prompt
        tokens followed by the generated completion.
    """
    tokens = torch.tensor(prompt, dtype=torch.long, device=model.device).unsqueeze(0)
    for _ in range(max_tokens):
        input_tokens = tokens[:, -context_length:]
        logits = model(input_tokens)
        next_logits = logits[..., -1, :]
        next_token = top_p_sampling(next_logits, p=p, temperature=temperature)
        
        tokens = torch.cat([tokens, next_token.unsqueeze(-1)], dim=-1)

        if (next_token == eos_token_id).all():
            break
    return tokens
