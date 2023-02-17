import torch

from ood_detection.models.dummy_zoc import CaptionGenerator


def sorted_zeroshot_weights(weights, split):
    sorted_weights = []
    for classname in split:
        sorted_weights.append(weights[classname])
    return torch.stack(sorted_weights).to(torch.float32)


def clean_caption(caption, classnames, truncate):
    result = [word for word in caption.split() if word not in classnames][:truncate]
    while len(result) < truncate:
        result.append("pad")
        print(f"Padded {caption}")
    return result


def generate_caption_with_generator(generator: CaptionGenerator, images_features, classnames, cut_off):
    captions = [generator.generate_caption(img_feat, True) for img_feat in images_features]
    return [clean_caption(caption, classnames, cut_off) for caption in captions]


def load_features_labels_captions(features_path, labels_path, captions_path):
    if torch.cuda.is_available():
        features = torch.load(features_path)
        labels = torch.load(labels_path)
    else:
        features = torch.load(features_path, map_location=torch.device('cpu'))
        labels = torch.load(labels_path, map_location=torch.device('cpu'))

    with open(captions_path, 'r', encoding='utf-8') as f:
        full_captions = [caption.rstrip('\n').split(" ") for caption in f]
    print("Loaded everything")
    return features, labels, full_captions


def save_features_labels_captions(features, features_path, labels, labels_path, captions, captions_path):
    torch.save(features, features_path)
    torch.save(labels, labels_path)
    with open(captions_path, 'w', encoding='utf-8') as f:
        for c in captions:
            f.write(" ".join(c) + '\n')
    print(f"Saved everything")


