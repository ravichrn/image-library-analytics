from PIL import Image

_MODEL_ID = "microsoft/Florence-2-base"

QUESTIONS = {
    "has_person": "Is there a person in this photo?",
    "setting": "Is this photo taken indoors or outdoors?",
    "time_of_day": "What time of day is shown in this photo?",
    "weather": "What is the weather like in this photo?",
    "season": "What season does this photo appear to be taken in?",
}
_KEYS = list(QUESTIONS.keys())
_TEXTS = list(QUESTIONS.values())


def load_caption_model(device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(_MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        _MODEL_ID,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()
    return model, processor


def extract_caption_batch(paths: list, model, processor, batch_size: int = 8) -> list[dict]:
    """
    For each batch of images, runs one forward pass per task (6 total):
    one for detailed caption + one per VQA question. Returns one dict per path.
    """
    import torch

    device = next(model.parameters()).device
    results: list[dict] = [{}] * len(paths)

    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        imgs: list = []
        valid_idx: list[int] = []
        for rel_i, path in enumerate(batch_paths):
            try:
                imgs.append(Image.open(str(path)).convert("RGB"))
                valid_idx.append(start + rel_i)
            except Exception:
                pass

        if not imgs:
            continue

        batch_data: list[dict] = [{} for _ in imgs]

        def _run(task_token: str, suffix: str = "", max_tokens: int = 64) -> list[str]:
            texts = [task_token + suffix] * len(imgs)
            inputs = processor(text=texts, images=imgs, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                out = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    attention_mask=inputs.get("attention_mask"),
                    max_new_tokens=max_tokens,
                    num_beams=3,
                    early_stopping=True,
                )
            decoded = processor.batch_decode(out, skip_special_tokens=False)
            return [
                processor.post_process_generation(d, task=task_token, image_size=(img.width, img.height)).get(task_token, "").strip()
                for d, img in zip(decoded, imgs, strict=False)
            ]

        try:
            for i, cap in enumerate(_run("<MORE_DETAILED_CAPTION>", max_tokens=128)):
                batch_data[i]["detailed_caption"] = cap
        except Exception:
            pass

        for key, question in QUESTIONS.items():
            try:
                answers = _run("<VQA>", question, max_tokens=16)
                for i, ans in enumerate(answers):
                    batch_data[i][key] = ans.lower()
            except Exception:
                pass

        for out_i, d in zip(valid_idx, batch_data, strict=False):
            results[out_i] = d

    return results
