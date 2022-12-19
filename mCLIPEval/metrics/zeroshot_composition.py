from tqdm import tqdm
import torch
from contextlib import suppress
def zeroshot_composition(model, dataset, batch_size, num_workers, verbose=False, amp=True):
    autocast = torch.cuda.amp.autocast if amp else suppress
    dataloader = dataset.get_dataloader(batch_size=batch_size, num_workers=num_workers, image_processor=model.image_processor)
    compositional_scores = []
    for images, texts in tqdm(dataloader):
        assert isinstance(images, list)
        assert isinstance(texts, list)

        texts0 = model.text_processor(texts=list(texts[0]))
        texts1 = model.text_processor(texts=list(texts[1]))
        with torch.no_grad(), autocast():
            text0_features = model.get_text_features(texts=texts0)
            text1_features = model.get_text_features(texts=texts1)
            image0_features = model.get_image_features(images=images[0])
            image1_features = model.get_image_features(images=images[1])
        image0_features = torch.tensor(image0_features.cpu().tolist())
        image1_features = torch.tensor(image1_features.cpu().tolist())
        text0_features = torch.tensor(text0_features.cpu().tolist())
        text1_features = torch.tensor(text1_features.cpu().tolist())

        c0_i0 = torch.nn.functional.cosine_similarity(image0_features, text0_features, dim = -1)
        c1_i0 = torch.nn.functional.cosine_similarity(image0_features, text1_features, dim = -1)
        c0_i1 = torch.nn.functional.cosine_similarity(image1_features, text0_features, dim = -1)
        c1_i1 = torch.nn.functional.cosine_similarity(image1_features, text1_features, dim = -1)
        compositional_score = torch.stack((c0_i0, c1_i0, c0_i1, c1_i1), dim=-1)
        compositional_scores.append(compositional_score)
    compositional_scores = torch.cat(compositional_scores)
    text_score, image_score, group_score = composition_metrics(example_results=compositional_scores.cpu().tolist())
    return {"text_score": text_score, "image_score": image_score, "group_score": group_score} 
    

def composition_metrics(example_results):
    def text_correct(example_result):
        return example_result[0]>example_result[1] and example_result[3]>example_result[2]
    def image_correct(example_result):
        return example_result[0]>example_result[2] and example_result[3]>example_result[1]
    def group_correct(example_result):
        return text_correct(example_result) and image_correct(example_result)
    
    text_correct_cnt = 0
    image_correct_cnt = 0
    group_correct_cnt = 0

    for result in example_results:
        assert isinstance(result, list) and len(result) == 4
        text_correct_cnt += 1 if text_correct(result) else 0
        image_correct_cnt += 1 if image_correct(result) else 0
        group_correct_cnt += 1 if group_correct(result) else 0
    
    denominator = len(example_results) * 1.0

    text_score = text_correct_cnt/denominator
    image_score = image_correct_cnt/denominator
    group_score = group_correct_cnt/denominator

    return text_score, image_score, group_score
