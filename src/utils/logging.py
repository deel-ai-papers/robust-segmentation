import torch


def brief_epoch_log(e, E, loss, m, show_cls=False):
    acc = float(m.get("acc", 0))
    miou = float(m.get("mIoU", 0))
    lip = float(m.get("lip", 0))
    rob = m.get("rob", {}) or {}
    rob_str = " ".join(
        f"{k.split('@')[-1]}={float(v):.2f}" for k, v in sorted(rob.items())
    )
    line = f"[{e}/{E}] loss={loss:.4f} acc={acc:.2%} mIoU={miou:.2%} lip:{lip:.2f} rob{{{rob_str}}}"
    if show_cls and (ci := m.get("cls_IoUs", None)) is not None:
        ci = (
            ci.detach().flatten().cpu()
            if torch.is_tensor(ci)
            else torch.tensor(ci).flatten()
        )
        nz = int((ci > 0).sum().item())
        topv, topi = torch.topk(ci, k=min(3, ci.numel()))
        line += (
            " "
            + f"cls(nz={nz}/{ci.numel()} top3="
            + ",".join(f"c{int(i)}={float(v):.3f}" for v, i in zip(topv, topi))
            + ")"
        )
    print(line)
