import argparse
from pathlib import Path

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, CLIPImageProcessor

from llavamini.train.llava_trainer import LLaVATrainer
from llavamini.train.train import (
    DataArguments,
    DataCollatorForSupervisedDataset,
    LazySupervisedDataset,
    ModelArguments,
    TrainingArguments,
)
from llavamini.model.language_model.llavamini_llama import LlavaMiniLlamaForCausalLM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--group-by-modality-length", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    model_path = root / "checkpoints/hf/ICTNLP/llava-mini-llama-3.1-8b"
    vision_path = root / "checkpoints/hf/openai/clip-vit-large-patch14-336"
    data_path = root / "data/smoke/train.json"
    image_folder = root / "data/smoke/images"
    output_dir = root / "checkpoints/runs/inspect_smoke_batch"

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    tokenizer.model_max_length = 1024

    image_processor = CLIPImageProcessor.from_pretrained(str(vision_path))

    data_args = DataArguments(
        data_paths=[str(data_path)],
        image_folders=[str(image_folder)],
        lazy_preprocess=True,
        is_multimodal=True,
        image_aspect_ratio="square",
    )
    data_args.image_processor = image_processor
    data_args.resolution_ratio = 1
    data_args.mm_use_im_start_end = False
    data_args.mm_use_im_patch_token = False

    dataset = LazySupervisedDataset(
        data_paths=data_args.data_paths,
        tokenizer=tokenizer,
        data_args=data_args,
    )
    collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    print("dataset_len", len(dataset))
    sample = dataset[0]
    print("sample_type", type(sample).__name__)
    print("sample_keys", sorted(sample.keys()))
    print("sample_input_ids_shape", tuple(sample["input_ids"].shape))
    print("sample_labels_shape", tuple(sample["labels"].shape))
    if "image" in sample:
        print("sample_image_shape", tuple(sample["image"].shape))

    batch = collator([sample])
    print("collator_batch_type", type(batch).__name__)
    print("collator_batch_keys", sorted(batch.keys()))
    print("collator_input_ids_shape", tuple(batch["input_ids"].shape))
    print("collator_labels_shape", tuple(batch["labels"].shape))
    if "images" in batch:
        print("collator_images_type", type(batch["images"]).__name__)
        print("collator_images_shape", tuple(batch["images"].shape))

    model_args = ModelArguments(
        model_name_or_path=str(model_path),
        version="llava_llama_3_1",
        vision_tower=str(vision_path),
        mm_projector_type="mlp2x_gelu",
        compressor_size=1,
        resolution_ratio=1,
        prefusion_layer_num=4,
        temporal_router_hidden_size=32,
        mm_vision_select_layer=-2,
        mm_vision_select_feature="patch",
        mm_use_im_start_end=False,
        mm_use_im_patch_token=False,
        tune_mm_mlp_adapter=True,
        freeze_backbone=True,
    )

    model = LlavaMiniLlamaForCausalLM.from_pretrained(
        str(model_path),
        cache_dir=None,
        attn_implementation=None,
        torch_dtype=None,
    )
    model.get_model().initialize_vision_modules(model_args=model_args, fsdp=None)
    vision_tower = model.get_vision_tower()
    vision_tower.image_processor = image_processor

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        max_steps=1,
        learning_rate=1e-4,
        logging_steps=1,
        save_steps=1,
        save_strategy="steps",
        evaluation_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        group_by_modality_length=args.group_by_modality_length,
        dataloader_num_workers=0,
        bf16=True,
        tf32=True,
    )

    trainer = LLaVATrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        data_collator=collator,
    )

    raw_loader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=collator,
        num_workers=0,
        drop_last=True,
    )
    raw_batch = next(iter(raw_loader))
    print("raw_batch_is_none", raw_batch is None)
    print("raw_batch_type", type(raw_batch).__name__)
    if raw_batch is not None:
        print("raw_batch_keys", sorted(raw_batch.keys()))

    dataloader = trainer.get_train_dataloader()
    first_batch = next(iter(dataloader))
    print("dataloader_batch_is_none", first_batch is None)
    print("dataloader_batch_type", type(first_batch).__name__)
    if first_batch is not None:
        print("dataloader_batch_keys", sorted(first_batch.keys()))
        for key, value in first_batch.items():
            if hasattr(value, "shape"):
                print(f"{key}_shape", tuple(value.shape))
            else:
                print(f"{key}_type", type(value).__name__)


if __name__ == "__main__":
    main()
