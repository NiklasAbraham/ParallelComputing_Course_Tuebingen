import torch


def cnn_forward_niklas(image, kernel, bias1, weights, bias2):
    # apply 2D convolution using unfold
    # input image shape: [16, 16]
    # kernel shape: [3, 3]
    # unfold both dimensions to create sliding windows
    # unfold along height: [16, 16] -> [14, 3, 16] (window size 3, step 1)
    # then unfold along width: [14, 3, 16] -> [14, 14, 3, 3] (window size 3, step 1)
    image_unfolded_h = image.unfold(0, 3, 1)  # [16, 16] -> [14, 3, 16]
    image_unfolded = image_unfolded_h.unfold(1, 3, 1)  # [14, 3, 16] -> [14, 14, 3, 3]
    # reshape to [14*14, 3*3] = [196, 9] for matrix multiplication
    image_patches = image_unfolded.reshape(-1, 3 * 3)  # [14, 14, 3, 3] -> [196, 9]
    kernel_flat = kernel.reshape(3 * 3)  # [3, 3] -> [9]
    # matrix multiplication: [196, 9] @ [9] -> [196] (broadcasting)
    # then reshape back to feature map: [196] -> [14, 14]
    feature_map = (image_patches @ kernel_flat).reshape(14, 14)  # [196] -> [14, 14]
    # add bias to feature map
    # bias1 is scalar, broadcasting: [14, 14] + scalar -> [14, 14]
    feature_map = feature_map + bias1
    # apply ReLU activation: element-wise maximum with zero
    # [14, 14] -> [14, 14]
    feature_map = torch.maximum(feature_map, torch.tensor(0.0))
    # flatten the feature map: [14, 14] -> [196]
    feature_map_flat = feature_map.reshape(196)  # [14, 14] -> [196]
    # linear layer: multiply by weight matrix and add bias
    # weights shape: [10, 196], feature_map_flat shape: [196]
    # matrix multiplication: [10, 196] @ [196] -> [10] (broadcasting)
    logits = weights @ feature_map_flat  # [10, 196] @ [196] -> [10]
    # add bias2: [10] + [10] -> [10]
    logits = logits + bias2
    return logits


def cnn_forward_filip(image, kernel, bias1, weights, bias2):
    # unfold image, changes shape from [16, 16] -> [14, 14, 3, 3]
    # last two dimensions hold the 3x3 matrix to apply the kernel two
    image_unfolded = image.unfold(0, 3, 1).unfold(1, 3, 1)

    # apply the kernel and sum up all values
    image_with_kernel  = (image_unfolded * kernel).sum(dim=(2,3))

    # add bias
    image_with_kernel_bias = image_with_kernel + bias1

    # ReLU activation
    image_relu = torch.maximum(image_with_kernel_bias, torch.tensor(0))

    # flatten image
    image_relu_flattened = torch.flatten(image_relu)

    # multiply flattened image with weight matrix + add bias2
    logits = torch.matmul(weights, image_relu_flattened) + bias2

    return logits

image = torch.randn(16, 16)
kernel = torch.randn(3, 3)
bias1 = torch.randn(1).item()
weights = torch.randn(10, 196)
bias2 = torch.randn(10)

logits_filip = cnn_forward_filip(image, kernel, bias1, weights, bias2)
logits_niklas = cnn_forward_niklas(image, kernel, bias1, weights, bias2)

print("Logits Filip:")
print(logits_filip)
print(f"\nPredicted digit Filip: {torch.argmax(logits_filip).item()}")

print("\nLogits Niklas:")
print(logits_niklas)
print(f"\nPredicted digit Niklas: {torch.argmax(logits_niklas).item()}")
