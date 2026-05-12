from PIL import Image

_MODEL_ID = "Salesforce/blip-vqa-base"

QUESTIONS = {
    "has_person":  "Is there a person in this photo?",
    "setting":     "Is this photo taken indoors or outdoors?",
    "time_of_day": "What time of day is shown in this photo?",
    "weather":     "What is the weather like in this photo?",
    "season":      "What season does this photo appear to be taken in?",
}
_KEYS = list(QUESTIONS.keys())
_TEXTS = list(QUESTIONS.values())
_NQ = len(_KEYS)


def load_caption_model(device: str):
    import torch
    from transformers import BlipForQuestionAnswering, BlipProcessor
    processor = BlipProcessor.from_pretrained(_MODEL_ID)
    model = BlipForQuestionAnswering.from_pretrained(
        _MODEL_ID, torch_dtype=torch.float16 if device != "cpu" else torch.float32
    )
    model.to(device)
    model.eval()
    return model, processor


def extract_caption_batch(
    paths: list, model, processor, batch_size: int = 8
) -> list[dict]:
    """
    Process `batch_size` photos at once.
    Each photo gets all NQ questions batched in a single generate() call,
    then we move to the next photo in the batch.
    """
    import torch

    device = next(model.parameters()).device
    results = [{}] * len(paths)

    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start:start + batch_size]
        for rel_i, path in enumerate(batch_paths):
            abs_i = start + rel_i
            try:
                img = Image.open(str(path)).convert("RGB")
                # Repeat the same image NQ times so all questions go in one generate() call
                inputs = processor(
                    [img] * _NQ, _TEXTS,
                    return_tensors="pt", padding=True
                ).to(device)
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=8)
                results[abs_i] = {
                    key: processor.decode(out[i], skip_special_tokens=True).strip().lower()
                    for i, key in enumerate(_KEYS)
                }
            except Exception:
                results[abs_i] = {}

    return results
