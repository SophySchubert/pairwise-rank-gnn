"""Generates a document causal attention mask based on a document ID tensor"""

from typing import List, Union

import torch
from torch import Tensor
from torch.nn.attention.flex_attention import _mask_mod_signature, noop_mask
from attn_gym.masks.causal import causal_mask


def _offsets_to_doc_ids_tensor(offsets):
    device = offsets.device
    counts = offsets[1:] - offsets[:-1]
    return torch.repeat_interleave(
        torch.arange(len(counts), device=device, dtype=torch.int32), counts
    )


def length_to_offsets(lengths: List[int], device: Union[str, torch.device]) -> Tensor:
    """Converts a list of lengths to a list of offsets.

    Args:
        lengths: A list of lengths.

    """
    offsets = [0]
    offsets.extend(lengths)
    offsets = torch.tensor(offsets, device=device, dtype=torch.int32)
    offsets = torch.cumsum(offsets, dim=-1)
    return offsets


def generate_doc_mask_mod(mask_mod: _mask_mod_signature, offsets: Tensor) -> _mask_mod_signature:
    """Generates mask mods that apply to inputs to flex attention in the sequence stacked
    format.

    Args:
        mask_mod: The mask mod to apply to the documents
        offsets: This tensor should be of shape(num_documents + 1)
            this should contain the cumulative counts of document tokens.
            e.g. if you have 3 documents of length 2, 4, 3 then
            offsets = [0, 2, 6, 9]

    Note:
        What is the sequence stacked format? When assembling batches of inputs, we
        take multiple sequences and stack them together to form 1 large sequence. We then
        use masking to ensure that the attention scores are only applied to tokens within
        the same document.
    """
    document_id = torch.tensor([0,0,0,1,1,1,1,2,2,2,3,3,3])#_offsets_to_doc_ids_tensor(offsets)
    bin_count = torch.bincount(document_id)
    unique = torch.unique(document_id)
    def doc_mask_mod(b, h, q_idx, kv_idx):
        print(q_idx, kv_idx, document_id, bin_count, unique)
        # doc_id1 = document_id[0:6]
        # doc_id2 = document_id[7:]
        # q_idx1 = q_idx.value
        # for x in q_idx:
        #     print(x)
        dif_doc = ((document_id[q_idx] != document_id[kv_idx])
                   & (((document_id[q_idx] == unique[0]) & (document_id[kv_idx] == unique[1])) | ((document_id[q_idx] == unique[1]) & (document_id[kv_idx] == unique[0])))
                   | (((document_id[q_idx] == unique[2]) & (document_id[kv_idx] == unique[3])) | ((document_id[q_idx] == unique[3]) & (document_id[kv_idx] == unique[2])))
                   )
        same_doc = document_id[q_idx] == document_id[kv_idx]
        return dif_doc # same_doc

    return doc_mask_mod


def main(device: str = "cpu", causal: bool = True):
    """Visualize the attention scores of document causal mask mod.

    Args:
        device (str): Device to use for computation. Defaults to "cpu".
    """
    from attn_gym import visualize_attention_scores
    import random

    random.seed(0)

    def generate_random_lengths(total_length, num_documents):
        # Initialize all lengths to 1 to ensure each document has at least one token
        lengths = [1] * num_documents
        remaining_length = total_length - num_documents

        # Randomly distribute the remaining length
        for _ in range(remaining_length):
            index = random.randint(0, num_documents - 1)
            lengths[index] += 1

        return lengths

    max_seq_len, doc_count = 13, 3
    B, H, SEQ_LEN, HEAD_DIM = 1, 1, max_seq_len, 8

    lengths = generate_random_lengths(max_seq_len, doc_count)

    offsets = length_to_offsets(lengths, device)

    def make_tensor():
        return torch.ones(B, H, SEQ_LEN, HEAD_DIM, device=device)

    query, key = make_tensor(), make_tensor()
    # print(query)
    if causal:
        base_mask_mod = causal_mask
    else:
        base_mask_mod = noop_mask
    document_causal_mask = generate_doc_mask_mod(base_mask_mod, offsets)
    visualize_attention_scores(
        query,
        key,
        mask_mod=document_causal_mask,
        device=device,
        name="graph_ranking_mask",
    )


if __name__ == "__main__":
    main('cpu', False)
