import torch


def sample_next_token(config, logits):
    if config.sampling_method == "argmax":
        return torch.argmax(logits, dim=-1).reshape(1)
    else:
        # Temperature
        logits = logits /config.temperature
        # Top-k sampling
        if config.top_k is not None:
            _values, indices = torch.topk(
                logits,
                config.top_k,
                dim=-1
            )
            # Mask For -inf
            mask = torch.ones_like(logits, dtype=torch.bool)
            mask.scatter_(dim=-1, index=indices, value=False)
            logits[mask] = -torch.inf
        # Convert to probability
        probs = torch.softmax(logits, dim=-1)

        if config.top_p is not None:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumulative_probs > config.top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            sorted_probs[sorted_indices_to_remove] = 0.0
            sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
            # Sample from sorted probabilities
            sampled = torch.multinomial(sorted_probs, num_samples=1)
            # Convert sorted position -> original token ID
            next_token = torch.gather(
                sorted_indices,
                -1,
                sampled
            )
        else:
            # Sample next token
            next_token = torch.multinomial(probs, num_samples=1)

        return next_token


def apply_repetition_penalty(config, logits: torch.Tensor, generated_ids: torch.Tensor):
    penalty = config.repetition_penalty
    if penalty == 1.0:
        return logits
    token_ids = generated_ids[0].unique()
    token_logits = logits[token_ids]
    token_logits = torch.where(
        token_logits > 0,
        token_logits / penalty,
        token_logits * penalty
    )
    logits[token_ids] = token_logits
    return logits