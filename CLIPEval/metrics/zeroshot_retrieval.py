"""
Code adapted from https://github.com/LAION-AI/CLIP_benchmark/blob/main/clip_benchmark/metrics/zeroshot_retrieval.py
Thanks to the authors of clip_benchmark
"""
import torch
import torch.nn.functional as F

from contextlib import suppress

from tqdm import tqdm
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

def zeroshot_retrieval(model, dataset, batch_size, num_workers, amp=True, recall_k_list=[1,5,10], verbose=False):
    """
    Evaluate the model on the given dataset
    Parameters
    ----------
    
    model: torch.nn,Module
        CLIP-like model with `encode_image` and `encode_text`
    
    dataloader: torch.utils.data.Dataloader
        dataloader to use for evaluation
    tokenizer:
        text tokenizer, i.e. convert list of strings to torch.Tensor of integers
    
    device: cpu/cuda
    amp: whether to use automatic mixed precision
    recall_k_list: list of int
        recall@k k's to use
    
    Returns
    -------
    
    dict of retrieval metrics
    """
    # list of batch of images embedding
    batch_images_emb_list = []
    # list of batch of text embedding
    batch_texts_emb_list = []
    # for each text, we collect the corresponding image index, as each image can have multiple corresponding texts
    texts_image_index = []
    dataloader = dataset.get_dataloader(batch_size=batch_size, num_workers=num_workers, image_processor=model.image_processor)
    dataloader = dataloader_with_indices(dataloader)
    autocast = torch.cuda.amp.autocast if amp else suppress
    for batch_images, batch_texts, inds in tqdm(dataloader):
        # store the index of image for each text, flickr30k 1 img -> 5 cap, muge 1 cap -> k img (k>1)
        if isinstance(batch_texts[0], list):
            batch_texts_image_index = [ind for ind, texts in zip(inds, batch_texts) for text in texts] # flickr30k  etc.
            batch_texts_squeeze = [ text for i, texts in enumerate(batch_texts) for text in texts]
            batch_images_squeeze = batch_images
        elif isinstance(batch_images[0],list):
            batch_texts_image_index = [ind for ind, imgs in zip(inds, batch_images) for img in imgs] # muge etc.
            batch_texts_squeeze = batch_texts
            batch_images_squeeze = torch.stack([torch.from_numpy(img) for i, imgs in enumerate(batch_images) for img in imgs])

        batch_texts_tok = model.text_processor(batch_texts_squeeze)
        
        # compute the embedding of images and texts
        with torch.no_grad(), autocast():
            batch_texts_emb = F.normalize(model.get_text_features(batch_texts_tok), dim=-1)
            batch_images_emb = F.normalize(model.get_image_features(batch_images_squeeze), dim=-1)
        batch_images_emb_list.append(batch_images_emb.cpu())
        batch_texts_emb_list.append(batch_texts_emb.cpu())
        texts_image_index.extend(batch_texts_image_index)
    batch_size = len(batch_images_emb_list[0])

    # concatenate all embeddings
    images_emb = torch.cat(batch_images_emb_list)
    texts_emb = torch.cat(batch_texts_emb_list)
    # get the score for each text and image pair
    if isinstance(batch_texts[0],list):
        scores  = texts_emb.float() @ images_emb.float().t()
    if isinstance(batch_images[0],list):
        scores  = images_emb.float() @ texts_emb.float().t()
    # construct a the positive pair matrix, which tells whether each text-image pair is a positive or not
    positive_pairs = torch.zeros_like(scores, dtype=bool)
    positive_pairs[torch.arange(len(scores)), texts_image_index] = True
    # we transpose to correct direction, if it's a muge datasets, which one caption with many images.
    if isinstance(batch_images[0],list):
        scores = scores.t()
        positive_pairs = positive_pairs.t()
    metrics = {}
    for recall_k in recall_k_list:
        # print('recall_k: ', recall_k)
        # Note that recall_at_k computes **actual** recall i.e. nb_true_positive/nb_positives, where the number
        # of true positives, e.g. for text retrieval, is, for each image,  the number of retrieved texts matching that image among the top-k.
        # Also, the number of positives are the total number of texts matching the image in the dataset, as we have a set of captions
        # for each image, that number will be greater than 1 for text retrieval.
        # However, image/text retrieval recall@k, the way it is done in CLIP-like papers, is a bit different.
        # recall@k, in CLIP-like papers, is, for each image, either 1 or 0. It is 1 if atleast one text matches the image among the top-k.
        # so we can easily compute that using the actual recall, by checking whether there is at least one true positive,
        # which would be the case if the recall is greater than 0. One we compute the recal for each image (or text), we average
        # it over the dataset.
        metrics[f"IR@{recall_k}"] = (batchify(recall_at_k, scores, positive_pairs, batch_size, model.device, k=recall_k)>0).float().mean().item()
        metrics[f"TR@{recall_k}"] = (batchify(recall_at_k, scores.T, positive_pairs.T, batch_size, model.device, k=recall_k)>0).float().mean().item()
    metrics[f"MR"] = sum([ i for i in metrics.values()]) / len(metrics.values())
    return metrics

def dataloader_with_indices(dataloader):
    start = 0
    for x, y in dataloader:            
        if isinstance(y[0], list) and len(x[0].shape)>3:
            end = start + len(x[0])
        else:
            end = start + len(x)
        inds = torch.arange(start, end)
        yield x, y, inds
        start = end
    

def recall_at_k(scores, positive_pairs, k):
    """
    Compute the recall at k for each sample
    :param scores: compability score between  text and image embeddings (nb texts, nb images)
    :param k: number of images to consider per text, for retrieval
    :param positive_pairs: boolean matrix of positive pairs (nb texts, nb images)
    :return: recall at k averaged over all texts
    """
    nb_texts, nb_images = scores.shape
    # for each text, sort according to image scores in decreasing order
    topk_indices = torch.topk(scores, k, dim=1)[1]
    # compute number of positives for each text
    nb_positive = positive_pairs.sum(dim=1)
    # nb_texts, k, nb_images
    topk_indices_onehot = torch.nn.functional.one_hot(topk_indices, num_classes=nb_images)
    # compute number of true positives
    positive_pairs_reshaped = positive_pairs.view(nb_texts, 1, nb_images)
    # a true positive means a positive among the topk
    nb_true_positive = (topk_indices_onehot * positive_pairs_reshaped).sum(dim=(1,2))
    # compute recall at k
    recall_at_k = (nb_true_positive / nb_positive)
    return recall_at_k

def batchify(func, X, Y, batch_size, device, *args, **kwargs):
    results = []
    for start in range(0, len(X), batch_size):
        end = start + batch_size
        x = X[start:end].to(device)
        y = Y[start:end].to(device)
        result = func(x, y, *args, **kwargs).cpu()
        results.append(result)
    return torch.cat(results)