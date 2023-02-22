import torch


def get_cosine_similarity_matrix_for_normed_features(image_features, text_features, temperature):
    return (temperature * image_features.to(torch.float32) @ text_features.T.to(torch.float32)).cpu()
