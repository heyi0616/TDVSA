import torch
from torch import nn
import torch.nn.functional as F

from data import consts


def greedy_decode(model, tgt_embed_ins, memory, max_len):
    """
    :param model: transformer_benchmark.Transformer
    :param tgt_embed_ins: target TokenEmbedding
    :param memory: (seq_len, 1, embed_size)
    :param max_len:
    :return: (n, 1)
    """
    ys = torch.full((1, 1), consts.BOS).type(torch.long).cuda()
    for i in range(max_len - 1):
        memory = memory.cuda()
        tgt_mask = (generate_square_subsequent_mask(ys.size(0)).type(torch.bool)).cuda()
        tgt_embed = tgt_embed_ins(ys)
        out = model.decode(tgt_embed, memory, tgt_mask)
        out = out.reshape(-1, out.shape[2])  # (seq_len, embed_size)
        prob = torch.matmul(out[-1, :], torch.transpose(tgt_embed_ins.weight, 0, 1))  # (vocab_size)

        next_word = torch.argmax(prob, -1).squeeze().item()
        ys = torch.cat([ys, torch.ones(1, 1).type(torch.long).fill_(next_word).cuda()], dim=0)
        if next_word == consts.EOS:
            break
    # print("greedy out size: " + str(ys.squeeze().shape))
    return ys.squeeze()


def beam_decode(model, tgt_embed_ins, memory, max_len):
    beam_size = 4
    alpha = 0.6
    blank_seqs = torch.full((max_len, beam_size), consts.PAD, dtype=torch.long).cuda()

    ys = torch.full((1, 1), consts.BOS).type(torch.long).cuda()
    tgt_mask = (generate_square_subsequent_mask(ys.size(0)).type(torch.bool)).cuda()
    tgt_embed = tgt_embed_ins(ys)
    dec_output = model.decode(tgt_embed, memory, tgt_mask)  # (seq_len, batch_size, embed_size)
    dec_output = torch.matmul(dec_output[-1, :, :], torch.transpose(tgt_embed_ins.weight, 0, 1))
    best_k_probs, best_k_idx = dec_output.topk(beam_size)  # (1, beam_size)
    # scores = torch.log(best_k_probs).view(beam_size)
    scores = F.log_softmax(best_k_probs, dim=-1).view(beam_size)
    gen_seq = blank_seqs
    gen_seq[0, :] = consts.BOS
    gen_seq[1, :] = best_k_idx[0]  # (seq_len, beam_size)
    enc_output = memory.repeat(1, beam_size, 1)  # (src_seq_len, beam_size, embed_size)
    ans_idx = 0  # default
    len_map = torch.arange(1, max_len + 1, dtype=torch.long).unsqueeze(1).cuda()
    for step in range(2, max_len):  # decode up to max
        tgt_mask = (generate_square_subsequent_mask(step).type(torch.bool)).cuda()
        # pad_mask = (gen_seq[:step, :] == consts.PAD).transpose(0, 1)
        pad_mask = None
        dec_output = model.decode(tgt_embed_ins(gen_seq[:step, :]), enc_output, tgt_mask,
                                  tgt_key_padding_mask=pad_mask)
        dec_output = torch.matmul(dec_output[-1, :, :], torch.transpose(tgt_embed_ins.weight, 0, 1))
        gen_seq, scores = _get_the_best_score_and_idx(gen_seq, dec_output, scores, step, beam_size)
        eos_locs = gen_seq == consts.EOS
        # -- replace the eos with its position for the length penalty use
        seq_lens, _ = len_map.masked_fill(~eos_locs, max_len).min(0)  # seq_lens: (beam_size)
        # -- check if all beams contain eos
        if (eos_locs.sum(0) > 0).sum(0).item() == beam_size:
            _, ans_idx = scores.div(seq_lens.float() ** alpha).max(0)
            ans_idx = ans_idx.item()
            break

    return gen_seq[:seq_lens[ans_idx], ans_idx]


def _get_the_best_score_and_idx(gen_seq, dec_output, scores, step, beam_size):
    assert len(scores.size()) == 1

    # Get k candidates for each beam, k^2 candidates in total.
    best_k2_probs, best_k2_idx = dec_output.topk(beam_size)

    # Include the previous scores.
    # scores = torch.log(best_k2_probs).view(beam_size, -1) + scores.view(beam_size, 1) # (beam_size, beam_size)
    predict_score = F.log_softmax(best_k2_probs, dim=-1).view(beam_size, -1)

    scores = predict_score + scores.view(beam_size, 1)
    # Get the best k candidates from k^2 candidates.
    scores, best_k_idx_in_k2 = scores.view(-1).topk(beam_size)

    # Get the corresponding positions of the best k candidiates.
    best_k_r_idxs, best_k_c_idxs = best_k_idx_in_k2 // beam_size, best_k_idx_in_k2 % beam_size
    best_k_idx = best_k2_idx[best_k_r_idxs, best_k_c_idxs]

    # Copy the corresponding previous tokens.
    gen_seq[:step, :] = gen_seq[:step, best_k_r_idxs]
    # Set the best tokens in this beam search step
    gen_seq[step, :] = best_k_idx

    return gen_seq, scores


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).cuda()
    return mask
