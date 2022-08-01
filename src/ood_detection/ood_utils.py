import clip
import torch

from ood_detection.classnames import fgvcaircraft_classes, caltech101_classes, oxfordpets_classes, flowers_classes, \
    dtd_classes, stanfordcars_classes
from ood_detection.config import Config
from ood_detection.models.dummy_zoc import CaptionGenerator


def classify(features, zeroshot_weights, labels, dataset):
    top1, top5, n = 0., 0., 0.
    logits = 100. * features @ zeroshot_weights
    acc1, acc5 = accuracy(logits, labels, top_k=(1, 5))
    top1 += acc1
    top5 += acc5
    n = features.size(0)
    top1 = (top1 / n) * 100
    top5 = (top5 / n) * 100
    print(f"\nClip Top1 Acc: {top1:.2f} with zeroshot on {dataset} ({features.size(0)} images)")
    print(f"\nClip Top5 Acc: {top5:.2f} with zeroshot on {dataset}")

    return top1


def accuracy(output, target, top_k=(1,)):
    pred = output.topk(max(top_k), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in top_k]


def ood_accuracy(output, target, num_id_labels=5, top_k=(1,)):
    pred = output.topk(max(top_k), 1, True, True)[1].t()
    for i, tensor in enumerate(pred):
        for j, value in enumerate(tensor):
            if value > num_id_labels - 1:
                pred[i, j] = num_id_labels

    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in top_k]


def get_normed_embeddings(classname, clip_model, templates):
    texts = [template.format(classname) for template in templates]
    texts = clip.tokenize(texts).to(Config.DEVICE)
    # casual normalization stuff, stolen from tip adapter paper
    class_embeddings = clip_model.encode_text(texts).to(torch.float32)  # embed
    class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
    class_embeddings = class_embeddings.mean(dim=0)
    class_embeddings /= class_embeddings.norm()
    return class_embeddings


def zeroshot_classifier(classnames: list, templates: list, clip_model):
    with torch.no_grad():
        weights = []
        for classname in classnames:
            class_embeddings = get_normed_embeddings(classname, clip_model, templates)
            weights.append(class_embeddings)

        weights = torch.stack(weights, dim=1)
        return weights


def clean_caption(caption, classnames, truncate):
    return [word for word in caption.split() if word not in classnames][:truncate]


def generate_caption_with_generator(generator: CaptionGenerator, images_features, classnames, cut_off):
    captions = [generator.generate_caption(img_feat, True) for img_feat in images_features]
    return [clean_caption(caption, classnames, cut_off) for caption in captions]


def get_full_logits(features, zeroshot_weights, captions, clip_model, templates):
    full_logits = []
    for i, (image, ood_labels) in enumerate(zip(features, captions)):
        with torch.no_grad():
            ind_class_embeddings = get_individual_ood_weights(ood_labels,
                                                              clip_model,
                                                              templates=templates)

            image = image.to(Config.DEVICE)
            ind_zeroshot_weights = torch.cat([zeroshot_weights, ind_class_embeddings], dim=1).to(Config.DEVICE)
            # zeroshotting
            logits = 100. * image @ ind_zeroshot_weights
            full_logits.append(logits)

    return torch.stack(full_logits)


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


def get_individual_ood_weights(caption, clip_model, templates):
    embeddings = []
    for word in caption:
        word_embedding = get_normed_embeddings(word, clip_model, templates)
        embeddings.append(word_embedding)

    embeddings = torch.stack(embeddings, dim=1)
    return embeddings


def get_ood_targets_from_sets(ood_names, clip_model, templates):
    # collect label embeddings from another dataset plan and simple, maybe just a random vector in clip space
    classnames = []
    for ood_name in ood_names:
        if ood_name == "fgvcaircraft":
            print(f"Using {ood_name} as ood")
            classnames.extend(fgvcaircraft_classes)

        if ood_name == "stanfordcars":
            print(f"Using {ood_name} as ood")
            classnames.extend(stanfordcars_classes)

        if ood_name == 'caltech101':
            print(f"Using {ood_name} as ood")
            classnames.extend(caltech101_classes)

        if ood_name == 'oxfordpets':
            print(f"Using {ood_name} as ood")
            classnames.extend(oxfordpets_classes)

        if ood_name == 'flowers':
            print(f"Using {ood_name} as ood")
            classnames.extend(flowers_classes)

        if ood_name == 'dtd':
            print(f"Using {ood_name} as ood")
            classnames.extend(dtd_classes)
    print(f'Number of classes merged to OOD label: {len(classnames)} from {len(ood_names)} sets')
    embedding = []
    for classname in classnames:
        class_embeddings = get_normed_embeddings(classname, clip_model, templates)
        embedding.append(class_embeddings)
    # take the center of this OOD
    embedding = torch.stack(embedding)
    embedding /= embedding.norm(dim=-1, keepdim=True)
    embedding = embedding.mean(dim=0)
    embedding /= embedding.norm()
    return embedding
