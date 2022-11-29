import clip
import torch
from datasets.classnames import imagenet_templates
from ood_detection.config import Config
from torch.utils.data import DataLoader
from tqdm import tqdm

device = Config.DEVICE


def full_classification(dataset, model, name):
    dataloader = DataLoader(dataset, batch_size=512)
    features, targets = get_dataset_features(dataloader, model, None, None)
    zeroshot_weights = zeroshot_classifier(dataset.classes, templates=imagenet_templates, clip_model=model)
    classify(features, zeroshot_weights, targets, name, True)


def get_dataset_features(loader: torch.utils.data.DataLoader, model, features_path, targets_path):
    features = []
    labels = []
    with torch.no_grad():
        for i, (images, target) in enumerate(tqdm(loader)):
            images = images.to(device)
            target = target.to(device)
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            features.append(image_features)
            labels.append(target)
        features = torch.cat(features)

        labels = torch.cat(labels)

        if features_path and targets_path:
            torch.save(features, features_path)
            torch.save(labels, targets_path)
    return features, labels


def zeroshot_classifier(classnames: list, templates: list, clip_model):
    with torch.no_grad():
        weights = []
        for classname in classnames:
            class_embeddings = get_normed_classname_embedding(classname, clip_model, templates)
            weights.append(class_embeddings)

        return torch.stack(weights)


def classify(features, zeroshot_weights, targets, dataset=None, print_results=False):
    top1, top5, n = 0., 0., 0.
    logits = 100. * features.to(torch.float32) @ zeroshot_weights.t().to(torch.float32)
    acc1, acc5 = accuracy(logits, targets, top_k=(1, 5))
    top1 += acc1
    top5 += acc5
    n = features.size(0)
    top1 = (top1 / n) * 100
    top5 = (top5 / n) * 100
    if print_results and dataset:
        print(f"\nClip Top1 Acc: {top1:.2f} with zeroshot on {dataset} ({features.size(0)} images)")
        print(f"\nClip Top5 Acc: {top5:.2f} with zeroshot on {dataset}")

    return top1, top5


def accuracy(output, target, top_k=(1,)):
    output = output.cpu()
    target = target.cpu()
    pred = output.topk(max(top_k), 1, True, True)[1].t()

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in top_k]


def get_normed_classname_embedding(classname, clip_model, templates):
    texts = [template.format(classname) for template in templates]
    texts = clip.tokenize(texts).to(Config.DEVICE)
    # casual normalization stuff, stolen from tip adapter paper
    with torch.no_grad():
        class_embeddings = clip_model.encode_text(texts).to(torch.float32)  # embed
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        class_embeddings = class_embeddings.mean(dim=0)
        class_embeddings /= class_embeddings.norm()
    return class_embeddings
