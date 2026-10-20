def _research_send_bounded_dcp_chunks(
    *, pack_buffer, src_ptrs, dst_ptrs, src_indices, dst_indices,
    token_item_lens, pack_fn, group_fn, transfer_fn,
):
    """Reuse one registered per-worker buffer only AFTER synchronous completion.

    This is a default-off research path. Indices are already request-relative
    DCP-mapped physical rows; slicing must preserve each source/destination
    pair. Never retry or reuse the buffer after a nonzero transfer status.
    """
    if len(src_ptrs) != len(dst_ptrs) or len(src_ptrs) != len(token_item_lens):
        raise ValueError('Mapped DCP layer lengths disagree')
    if len(src_indices) != len(dst_indices):
        raise ValueError('Mapped DCP row lengths disagree')
    if not token_item_lens or any(int(n) <= 0 for n in token_item_lens):
        raise ValueError('Invalid DCP token item lengths')
    row_bytes = sum(int(n) for n in token_item_lens)
    max_rows = pack_buffer.get_size() // row_bytes
    if max_rows < 1:
        raise ValueError('DCP pack buffer cannot hold one full row')
    for begin in range(0, len(src_indices), max_rows):
        end = min(begin + max_rows, len(src_indices))
        packed = pack_fn(
            pack_buffer=pack_buffer, kv_data_ptrs=src_ptrs,
            src_token_indices=src_indices[begin:end],
            token_item_lens=token_item_lens,
        )
        if packed is None:
            raise RuntimeError('Bounded DCP chunk unexpectedly does not fit')
        packed_ptrs, packed_indices = packed
        src_groups, dst_groups = group_fn(packed_indices, dst_indices[begin:end])
        blocks = []
        for src_ptr, dst_ptr, item_len in zip(packed_ptrs, dst_ptrs, token_item_lens):
            for src_group, dst_group in zip(src_groups, dst_groups):
                blocks.append((
                    src_ptr + int(src_group[0]) * int(item_len),
                    dst_ptr + int(dst_group[0]) * int(item_len),
                    len(src_group) * int(item_len),
                ))
        status = transfer_fn(blocks)
        if status != 0:
            return status
    return 0
