import torch
import torch.nn.functional as F


class PeakInference:
    def __init__(
        self,
        model_class,          # 🔥 pass class, not instance
        model_config,         # config needed to init model
        weight_path,          # path to .pt / .pth
        tokenizer,
        device="cuda"
    ):
        self.device = device
        self.tokenizer = tokenizer

        # 🔥 initialize model
        self.model = model_class(**model_config).to(device)

        # 🔥 load weights (state_dict)
        checkpoint = torch.load(weight_path, map_location=device)

        # support both formats
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        prompt,
        max_new_tokens=50,
        temperature=1.0,
        top_k=None,
        top_p=None,
        eos_token_id=None
    )->str:
        # 🔥 encode properly
        prompt_ids = self.tokenizer.encode(prompt)
        prompt_len = len(prompt_ids)
        input_ids = torch.tensor(
            [prompt_ids],
            dtype=torch.long,
            device=self.device
        )

        for _ in range(max_new_tokens):
            logits = self.model(input_ids)
            logits = logits[:, -1, :]

            # temperature
            if temperature != 1.0:
                logits = logits / temperature

            probs = F.softmax(logits, dim=-1)

            # 🔹 top-k
            if top_k is not None:
                top_k_vals, top_k_idx = torch.topk(probs, top_k)
                probs = torch.zeros_like(probs).scatter_(1, top_k_idx, top_k_vals)
                probs = probs / probs.sum(dim=-1, keepdim=True)

            # 🔹 top-p (nucleus)
            if top_p is not None:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                mask = cumulative_probs > top_p
                mask[:, 1:] = mask[:, :-1].clone()
                mask[:, 0] = False

                sorted_probs[mask] = 0.0
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

                probs = torch.zeros_like(probs).scatter_(1, sorted_idx, sorted_probs)

            # sample
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # stop
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

        output_ids = input_ids[0].tolist()[prompt_len:]
        return self.tokenizer.decode(output_ids).strip()