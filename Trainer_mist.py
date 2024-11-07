print("Starting")
import pathlib
print("Loaded pathlib")
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
#from transformers import VivitImageProcessor, VivitForVideoClassification
print("Loaded transformers")
import pytorchvideo.data
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    UniformTemporalSubsample,
)
print("Loaded pytorchvideo")
from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
    Resize,
)

import os
import numpy as np
from transformers import TrainingArguments, Trainer
#import transformers
print("Loaded transformers Trainer")
import evaluate
print("Loaded evaluate")
import torch
import yaml
import torch.nn as nn
print("Loaded pytorch")

print("Cuda Version:" + str(torch.version.cuda))
metric = evaluate.load("accuracy")
# metric = evaluate.load(os.path.join(os.environ.get('SCRATCH'), 'huggingface', 'metrics', 'accuracy', 'default', 'accuracy.py'))
print("Loaded Metric")


def collate_fn(examples):
    # permute to (num_frames, num_channels, height, width)
    pixel_values = torch.stack(
        [example["video"].permute(1, 0, 2, 3) for example in examples]
    )
    labels = torch.tensor([example["label"] for example in examples])
    output = {"pixel_values": pixel_values, "labels": labels}
    print(pixel_values.shape)
    print(labels)
    print(output)
    return output


def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)


def unnormalize_img(img):
    """Un-normalizes the image pixels."""

    img = (img * std) + mean
    img = (img * 255).astype("uint8")
    return img.clip(0, 255)


def run_inference(model, video):
    # (num_frames, num_channels, height, width)
    perumuted_sample_test_video = video.permute(1, 0, 2, 3)
    inputs = {
        "pixel_values": perumuted_sample_test_video.unsqueeze(0)
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model = model.to(device)
    # forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    return logits


with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

train = config['train']
#transformers.logging.set_verbosity_level_info()
dataset_root_path = config['master_file_path'] + 'vids/'
dataset_root_path = pathlib.Path(dataset_root_path)

video_count_train = len(list(dataset_root_path.glob("train/*/*.avi")))
video_count_val = len(list(dataset_root_path.glob("val/*/*.avi")))
video_total = video_count_train + video_count_val
print(f"Total videos: {video_total}")

all_video_file_paths = (
        list(dataset_root_path.glob("train/*/*.avi"))
        + list(dataset_root_path.glob("val/*/*.avi"))
)

class_labels = sorted({str(path).split("/")[-2] for path in all_video_file_paths})
label2id = {label: i for i, label in enumerate(class_labels)}
id2label = {i: label for label, i in label2id.items()}


print(f"Unique classes: {list(label2id.keys())}.")

if train:

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Device: " + str(device))

    # MCG-NJU/videomae-base-finetuned-kinetics.
    model_ckpt = config['model_name']

    # Video MAE
    image_processor = VideoMAEImageProcessor.from_pretrained(model_ckpt)
    model = VideoMAEForVideoClassification.from_pretrained(model_ckpt, label2id=label2id, id2label=id2label,
                                                           ignore_mismatched_sizes=True)

    # ViViT
    # image_processor = VivitImageProcessor.from_pretrained(model_ckpt)
    # model = VivitForVideoClassification.from_pretrained(model_ckpt)
    # model.classifier = nn.Linear(model.classifier.in_features, 2)

    # If using checkpoint
    # model_name = model_ckpt.split("/")[-1]num_gpus = torch.cuda.device_count()
    # model = f"{model_name}-finetuned-" + str(config['num_stacks'])

    mean = image_processor.image_mean
    std = image_processor.image_std

    if "shortest_edge" in image_processor.size:
        height = width = image_processor.size["shortest_edge"]
    else:
        height = image_processor.size["height"]
        width = image_processor.size["width"]

    resize_to = (height, width)

    num_frames_to_sample = model.config.num_frames
    print("Number of frames accepted by model: " + str(num_frames_to_sample))
    sample_rate = config['sample_rate']
    fps = config['fps']
    clip_duration = num_frames_to_sample * sample_rate / fps

    train_transform = Compose(
        [
            ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        UniformTemporalSubsample(num_frames_to_sample),
                        Lambda(lambda x: x / 255.0),
                        Normalize(mean, std),
                        # RandomShortSideScale(min_size=config['random_sss_minsize'], max_size=config['random_sss_maxsize']),
                        # RandomCrop(resize_to),
                        # RandomHorizontalFlip(p=config['random_hf_p']),
                    ]
                ),
            ),
        ]
    )

    train_dataset = pytorchvideo.data.Ucf101(
        data_path=os.path.join(dataset_root_path, "train"),
        clip_sampler=pytorchvideo.data.make_clip_sampler("random", clip_duration),
        decode_audio=False,
        transform=train_transform,
    )

    val_transform = Compose(
        [
            ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        UniformTemporalSubsample(num_frames_to_sample),
                        Lambda(lambda x: x / 255.0),
                        Normalize(mean, std),
                        Resize(resize_to),
                    ]
                ),
            ),
        ]
    )

    val_dataset = pytorchvideo.data.Ucf101(
        data_path=os.path.join(dataset_root_path, "val"),
        clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
        decode_audio=False,
        transform=val_transform,
    )

    print("Training Videos: " + str(train_dataset.num_videos))
    print("Validation Videos: " + str(val_dataset.num_videos))

    num_epochs = config['num_epochs']
    batch_size = config['batch_size']
    model_name = model_ckpt.split("/")[-1]
    new_model_name = f"{model_name}-finetuned-" + str(video_total)
    args = TrainingArguments(
        new_model_name,
        remove_unused_columns=config['remove_unused_columns'],
        evaluation_strategy=config['eval_strategy'],
        eval_steps=config['eval_steps'],
        save_strategy=config['save_strategy'],
        save_steps=config['save_steps'],
        learning_rate=config['learning_rate'],
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_ratio=config['warmup_ratio'],
        fp16=True,
        logging_steps=config['logging_steps'],
        load_best_model_at_end=config['load_best_model_at_end'],
        metric_for_best_model=config['metric_for_best_model'],
        push_to_hub=config['push_to_hub'],
        ignore_data_skip=config['ignore_data_skip'],
        max_steps=(train_dataset.num_videos // batch_size // config['num_gpus']) * num_epochs,
    )

    num_gpus = args.n_gpu
    print(f"Number of GPUs being used (maybe incorrect): {num_gpus}")
    
    
    trainer = Trainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=image_processor,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )


    print(f"Model distributed parallel: {trainer.args.parallel_mode}")

    train_results = trainer.train(resume_from_checkpoint=config['resume_from_checkpoint'])

    for obj in trainer.state.log_history:
        print(obj)

else:

    checkpoint_path = os.path.join('videomae-base-finetuned-100', 'checkpoint-36')
    model = VideoMAEForVideoClassification.from_pretrained(checkpoint_path, local_files_only=True)
    image_processor = VideoMAEImageProcessor.from_pretrained(checkpoint_path)

    mean = image_processor.image_mean
    std = image_processor.image_std

    if "shortest_edge" in image_processor.size:
        height = width = image_processor.size["shortest_edge"]
    else:
        height = image_processor.size["height"]
        width = image_processor.size["width"]

    resize_to = (height, width)

    num_frames_to_sample = model.config.num_frames
    print(num_frames_to_sample)
    sample_rate = 1
    fps = 2
    clip_duration = num_frames_to_sample * sample_rate / fps

    test_transform = Compose(
        [
            ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        UniformTemporalSubsample(num_frames_to_sample),
                        Lambda(lambda x: x / 255.0),
                        Normalize(mean, std),
                        Resize(resize_to),
                    ]
                ),
            ),
        ]
    )

    test_dataset = pytorchvideo.data.Ucf101(
        data_path=os.path.join(dataset_root_path, "test"),
        clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
        decode_audio=False,
        transform=test_transform,
    )

    # Create a DataLoader
    dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,  # Adjust batch size as needed
        num_workers=4,  # Adjust number of workers as needed
    )

    for batch in dataloader:
        videos = batch['video']
        labels = batch['label']
        logits = run_inference(model, videos.squeeze())
        predicted_class_idx = logits.argmax(-1).item()
        print("Predicted class:", model.config.id2label[predicted_class_idx])
