import numpy as np
import torch


def generate_mask_patterns(K, seq_len, ratio):
    """Generate K fixed masking patterns."""
    n = int(seq_len * ratio)

    masks = []

    for _ in range(K):
        idx = torch.randperm(seq_len)[:n]

        mask = torch.zeros(
            seq_len,
            dtype=torch.long,
        )
        mask[idx] = 1

        masks.append(mask)

    return torch.stack(masks)


@torch.no_grad()
def corrupt_input_random(masked_ids, mask_token_id, vocab_size):
    """Replace masked tokens with randomly sampled vocabulary tokens."""
    corrupted = masked_ids.clone()
    mask_positions = masked_ids == mask_token_id

    n_to_replace = mask_positions.sum().item()

    if n_to_replace > 0:
        random_tokens = torch.randint(
            5,
            vocab_size,
            (n_to_replace,),
            device=masked_ids.device,
        )

        corrupted[mask_positions] = random_tokens

    return corrupted


@torch.no_grad()
def corrupt_input(
    generator,
    masked_ids,
    attention_mask,
    mask_token_id,
):
    """Replace masked tokens using samples from the generator."""
    logits = generator(
        masked_ids,
        attention_mask,
    )

    logits = torch.clamp(
        logits,
        min=-1e9,
        max=1e9,
    )

    probs = torch.softmax(
        logits,
        dim=-1,
    )

    sampled = torch.multinomial(
        probs.view(-1, probs.size(-1)),
        1,
    ).view(masked_ids.shape)

    corrupted = masked_ids.clone()
    mask_positions = masked_ids == mask_token_id
    corrupted[mask_positions] = sampled[mask_positions]

    return corrupted


def date_loss(
    rtd_logits,
    rmd_logits,
    rtd_labels,
    rmd_labels,
    mlm_logits,
    mlm_labels,
    mu=100.0,
    lambda_rtd=50.0,
):
    """Compute the combined DATE loss."""
    ce = torch.nn.CrossEntropyLoss(
        ignore_index=-100,
    )
    bce = torch.nn.BCEWithLogitsLoss(
        reduction="none",
    )

    loss_mlm = ce(
        mlm_logits.view(-1, mlm_logits.size(-1)),
        mlm_labels.view(-1),
    )

    valid = rtd_labels != -100

    if valid.sum() > 0:
        loss_rtd = bce(
            rtd_logits[valid].float(),
            rtd_labels[valid].float(),
        ).mean()
    else:
        loss_rtd = torch.tensor(
            0.0,
            device=rtd_logits.device,
        )

    loss_rmd = ce(
        rmd_logits,
        rmd_labels,
    )

    return (
        mu * loss_rmd
        + loss_mlm
        + lambda_rtd * loss_rtd
    )


def apply_mask_safe(input_ids, mask, tokenizer):
    """Apply a mask pattern while preserving special tokens."""
    batch_size, _ = input_ids.shape
    masked = input_ids.clone()

    mask_expanded = (
        mask.unsqueeze(0)
        .expand(batch_size, -1)
        .clone()
    )

    forbidden = torch.zeros_like(
        input_ids,
        dtype=torch.bool,
    )

    forbidden |= input_ids == tokenizer.cls_token_id
    forbidden |= input_ids == tokenizer.sep_token_id
    forbidden |= input_ids == tokenizer.pad_token_id

    final_mask = (
        (mask_expanded == 1)
        & (~forbidden)
    )

    masked[final_mask] = tokenizer.mask_token_id

    return masked, final_mask


@torch.no_grad()
def date_anomaly_score(
    discriminator,
    dataloader,
    tokenizer,
    device="cuda",
):
    """Compute PL_RTD anomaly scores for the input data."""
    discriminator.eval()

    all_scores = []
    all_labels = []

    for input_ids, attention_mask, labels in dataloader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        rtd_logits, _ = discriminator(
            input_ids,
            attention_mask,
        )

        p_replaced = torch.sigmoid(rtd_logits)
        p_original = 1.0 - p_replaced

        valid = (
            (attention_mask == 1)
            & (input_ids != tokenizer.cls_token_id)
            & (input_ids != tokenizer.pad_token_id)
        )

        seq_scores = (
            (p_original * valid).sum(dim=1)
            / valid.sum(dim=1).clamp(min=1)
        )

        all_scores.extend(
            seq_scores.cpu().numpy()
        )
        all_labels.extend(
            labels.numpy()
        )

    scores = np.array(all_scores)
    labels = np.array(all_labels)

    return scores, labels