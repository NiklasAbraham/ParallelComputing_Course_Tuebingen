import math

import torch

H = 4
N = 8
M = 16
D = 12


def attention(input, query_weights, key_weights, value_weights):
    # multi-head attention: compute H heads in parallel using broadcasting
    # input shape: [N, M]
    # query_weights shape: [H, M, D]
    # key_weights shape: [H, M, D]
    # value_weights shape: [H, M, D]
    # expand input from [N, M] -> [1, N, M] for broadcasting
    # then matrix multiplication: [1, N, M] @ [H, M, D] -> [H, N, D]
    input_expanded = input.unsqueeze(0)  # [N, M] -> [1, N, M]
    query = input_expanded @ query_weights  # [1, N, M] @ [H, M, D] -> [H, N, D]
    key = input_expanded @ key_weights  # [1, N, M] @ [H, M, D] -> [H, N, D]
    value = input_expanded @ value_weights  # [1, N, M] @ [H, M, D] -> [H, N, D]
    # compute attention scores for each head
    # transpose key from [H, N, D] -> [H, D, N] for batch matrix multiplication
    # query @ key.T: [H, N, D] @ [H, D, N] -> [H, N, N]
    key_transposed = key.transpose(1, 2)  # [H, N, D] -> [H, D, N]
    scores = query @ key_transposed  # [H, N, D] @ [H, D, N] -> [H, N, N]
    # scale by sqrt(D): [H, N, N] -> [H, N, N]
    scores = scores / math.sqrt(D)
    # apply softmax along the sequence dimension (dim=2) for each head
    # softmax: [H, N, N] -> [H, N, N]
    weights = torch.softmax(scores, dim=2)
    # compute output: attention weights @ value
    # [H, N, N] @ [H, N, D] -> [H, N, D]
    output = weights @ value
    return output


input = torch.randn(N, M)
query_weights = torch.randn(H, M, D)
key_weights = torch.randn(H, M, D)
value_weights = torch.randn(H, M, D)

output = attention(input, query_weights, key_weights, value_weights)

print("Output shape:", output.shape)
print("Output:")
print(output)
