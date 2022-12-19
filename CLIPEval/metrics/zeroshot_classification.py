"""
Code adapted from https://github.com/LAION-AI/CLIP_benchmark/blob/main/clip_benchmark/metrics/zeroshot_classification.py
Thanks to the authors of clip_benchmark
"""
import torch
import torch.nn.functional as F

from contextlib import suppress

from tqdm import tqdm
from sklearn.metrics import classification_report, balanced_accuracy_score
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

def zero_shot_classifier(model, dataset, amp=True):
    autocast = torch.cuda.amp.autocast if amp else suppress
    with torch.no_grad(), autocast():
        zeroshot_weights = []
        for classname in tqdm(dataset.classes):
            texts = [template.format(c=classname) for template in dataset.templates]
            texts = model.text_processor(texts=texts)
            class_embedding = model.get_text_features(texts=texts)
            class_embedding = F.normalize(class_embedding, dim=-1).mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1)
    return zeroshot_weights

def run_classification(model, dataset, batch_size, num_workers, amp=True):
    autocast = torch.cuda.amp.autocast if amp else suppress
    pred = []
    true = []
    nb = 0
    dataloader = dataset.get_dataloader(batch_size=batch_size, num_workers=num_workers, image_processor=model.image_processor)
    classifier = zero_shot_classifier(dataset=dataset, model=model, amp=amp)
    with torch.no_grad():
        for images, targets in tqdm(dataloader):
            with autocast():
                image_features = model.get_image_features(images)
                image_features = F.normalize(image_features, dim=-1)
                logits = 100. * image_features @ classifier
            true.append(targets.cpu())
            pred.append(logits.cpu())
    pred = torch.cat(pred)
    true = torch.cat(true)
    return pred, true

def accuracy(output, target, topk=(1,)):
    """
    Compute top-k accuracy
    output: torch.Tensor
        shape (N, C) where N is the number of examples, C the number of classes.
        these are the logits.
    
    target: torch.Tensor
        shape (N,) where N is the number of examples. Groundtruth class id of each example.
    
    topk: tuple
        which topk to compute, e.g., topk=(1,5) will compute top-1 and top-5 accuracies
    
    Returns
    -------
    
    list of top-k accuracies in the same order as `topk`
    """
    # pred = output.topk(max(topk), 1, True, True)[1].t()
    pred = output.float().topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    n = len(target)
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) / n for k in topk]

def average_precision_per_class(scores, targets):
    """
    Compute average precision  for each class
    this metric is used for multi-label classification
    see explanations here https://fangdahan.medium.com/calculate-mean-average-precision-map-for-multi-label-classification-b082679d31be
    Code is adapted from https://github.com/pytorch/tnt/blob/master/torchnet/meter/meter.py, thanks to the authors of `tnt`.
    Parameters
    ----------
    scores: torch.Tensor
        logits, of shape (N,C) where N is the number of examples, C the number of classes
    
    targets: torch.Tensor
        one-hot vectors of groundtruth targets (N, C), where N is the number of examples, C is the
        number of classes
    
    Returns
    -------
    torch.Tensor of shape (C,) of avereage precision for each class, where C is     
    the number of classes.
    
    """
    ap = torch.zeros(scores.size(1))
    rg = torch.arange(1, scores.size(0) + 1).float()
    # compute average precision for each class
    for k in range(scores.size(1)):
        # sort scores
        scores_k = scores[:, k]
        targets_k = targets[:, k]
        _, sortind = torch.sort(scores_k, 0, True)
        truth = targets_k[sortind]
        tp = truth.float().cumsum(0)
        # compute precision curve
        precision = tp.div(rg)
        # compute average precision
        ap[k] = precision[truth.bool()].sum() / max(float(truth.sum()), 1)
    return ap


def zeroshot_classification(model, dataset, batch_size, num_workers, verbose=False):
    logits, target = run_classification(model=model, dataset=dataset, batch_size=batch_size, num_workers=num_workers)
    is_multilabel = (len(target.shape) == 2)

    if is_multilabel:
        if verbose:
            print("Detected a multi-label classification dataset")
        # Multiple labels per image, multiple classes on the dataset
        ap_per_class = average_precision_per_class(logits, target)
        if verbose:
            for class_name, ap in zip(dataset.classes, ap_per_class.tolist()):
                print(f"Class: {class_name}, AveragePrecision: {ap}")
        return {"mean_average_precision": ap_per_class.mean().item()}
    
    else:
        pred = logits.argmax(axis=1)
        # measure accuracy
        if len(dataset.classes) >= 5:
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        else:
            acc1, = accuracy(logits, target, topk=(1,))
            acc5 = float("nan") 
        mean_per_class_recall = balanced_accuracy_score(target, pred)
        if verbose:
            print(classification_report(target, pred, digits=3))
        return {"acc1": acc1, "acc5": acc5, "mean_per_class_recall": mean_per_class_recall}
