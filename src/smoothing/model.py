import torch
import torch.nn as nn


class SmoothedModel(nn.Module):
    def __init__(self, model, sigma: float, n_mc: int, max_bs: int):
        super().__init__()
        self.model = model.eval()
        self.sigma = float(sigma)
        self.n_mc = int(n_mc)
        self.max_bs = int(max_bs)

    @torch.no_grad()
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # returns on CPU to avoid GPU blowup: [B, n_mc, H, W] uint16/uint8
        B, *rest = inputs.shape  # rest = [C,H,W] for images
        device, dtype = inputs.device, inputs.dtype
        H, W = rest[-2], rest[-1]

        # choose smallest dtype that fits classes
        out_dtype = torch.uint8  # if num_classes > 255 switch to torch.int16

        out_cpu = torch.empty((B, self.n_mc, H, W), dtype=out_dtype, device="cpu")
        write = 0
        for k in range(0, self.n_mc, self.max_bs):
            bs = min(self.max_bs, self.n_mc - k)
            noise = (
                torch.randn((bs, *inputs.shape[1:]), device=device, dtype=dtype)
                * self.sigma
            )
            noisy = (inputs.unsqueeze(0) + noise.unsqueeze(1)).reshape(
                bs * B, *inputs.shape[1:]
            )
            logits = self.model(noisy)
            if isinstance(logits, tuple):
                logits = logits[0] / logits[1]
            preds = (
                logits.argmax(dim=1).reshape(bs, B, H, W).permute(1, 0, 2, 3)
            )  # [B,bs,H,W]
            out_cpu[:, write : write + bs].copy_(
                preds.to(out_cpu.dtype, non_blocking=True).cpu()
            )
            write += bs
            del noise, noisy, logits, preds
            torch.cuda.empty_cache()  # optional
        return out_cpu  # CPU tensor
