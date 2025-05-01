import torch

import torch.nn.functional as F


def compute_ce_and_dur_losses(d, d_gt, input_lengths):
    loss_ce = 0
    loss_dur = 0

    for _s2s_pred, _text_input, _text_length in zip(d, d_gt, input_lengths):
        _s2s_pred = _s2s_pred[:_text_length, :]
        _text_input = _text_input[:_text_length].long()
        _s2s_trg = torch.zeros_like(_s2s_pred)
        for p in range(_s2s_trg.shape[0]):
            _s2s_trg[p, :_text_input[p]] = 1
        _dur_pred = torch.sigmoid(_s2s_pred).sum(axis=1)

        loss_dur += F.l1_loss(_dur_pred[1:_text_length - 1], _text_input[1:_text_length - 1])
        loss_ce += F.binary_cross_entropy_with_logits(_s2s_pred.flatten(), _s2s_trg.flatten())

    loss_ce /= d.size(0)
    loss_dur /= d.size(0)

    return loss_ce, loss_dur


def compute_s2s_loss(s2s_pred, texts, input_lengths):
    loss_s2s = 0

    for _s2s_pred, _text_input, _text_length in zip(s2s_pred, texts, input_lengths):
        loss_s2s += F.cross_entropy(_s2s_pred[:_text_length], _text_input[:_text_length])

    loss_s2s /= texts.size(0)

    return loss_s2s
