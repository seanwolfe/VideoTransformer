import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
# import cv2
from torch import nn
import torch.optim as optim
from transformers import VideoMAEForVideoClassification, Trainer, TrainingArguments, VideoMAEImageProcessor
import pandas as pd
import ast
import os
import yaml
from pytorchvideo.transforms import (
    Normalize,
)
import evaluate

print("Loaded pytorchvideo")
from torchvision.transforms import (
    Lambda
)

metric = evaluate.load("mae", "multilist")


# metric = evaluate.load(os.path.join(os.environ.get('SCRATCH'), 'huggingface', 'metrics', 'mae', 'mae.py'), "multilist")


# Define a converter function
def str_to_tuple(x):
    try:
        return ast.literal_eval(x)
    except (ValueError, SyntaxError):
        return x


def read_master(configuration, master_file_path):
    columns_to_convert = configuration['center_file_columns']
    columns_to_convert_final = ['Stack_Crop_Start'] + columns_to_convert
    return pd.read_csv(master_file_path, sep=',', header=0,
                       converters={col: str_to_tuple for col in columns_to_convert_final},
                       names=configuration['master_file_columns'])


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return metric.compute(predictions=predictions, references=labels)


def collate_fn(examples):
    pixel_values = torch.stack(
        [example['video'] for example in examples]
    ).to(torch.float32)
    labels = torch.stack([example['label'] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


# Custom Dataset Class
class VideoDataset(Dataset):
    def __init__(self, video_files, labels, transform=None):
        self.video_files = video_files
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float)

        # Load the video and extract frames
        video_array = np.load(video_path)
        video_tensor = torch.from_numpy(video_array)
        if self.transform:
            video_tensor = self.transform(video_tensor.permute(1, 0, 2, 3))
            video_tensor = video_tensor.permute(1, 0, 2, 3)
        res = {"video": video_tensor, "label": label}
        return res


with open('config_regressor.yaml', 'r') as f:
    config = yaml.safe_load(f)

if config['train']:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Device: " + str(device))

    # MCG-NJU/videomae-base-finetuned-kinetics.
    model_ckpt = config['model_name']

    # Load the VideoMAE model for regression
    image_processor = VideoMAEImageProcessor.from_pretrained(model_ckpt)
    model = VideoMAEForVideoClassification.from_pretrained(model_ckpt, num_labels=32)
    model.config.problem_type = "regression"
    model.loss_fct = nn.L1Loss()

    if config['mean']:
        mean = config['mean']
    else:
        mean = image_processor.image_mean  # for RGB videos typical values
    std = []
    if config['std']:
        std = config['std']
    else:
        std = image_processor.image_std

    # Data Transformations for both val and train
    transform = transforms.Compose([
        Lambda(lambda x: x / 255.0),
        Normalize(mean, std),

    ])

    # get  train master file
    master_file_path = os.path.join(config['data_path'], config['master_file_name_train'])
    master_file = read_master(config, master_file_path)

    # Example Video Files and Labels
    video_files = master_file['Saved_as_Stack']
    columns_to_transform = ['F1_Center', 'F2_Center', 'F3_Center', 'F4_Center', 'F5_Center', 'F6_Center', 'F7_Center',
                            'F8_Center', 'F9_Center', 'F10_Center', 'F11_Center', 'F12_Center', 'F13_Center',
                            'F14_Center', 'F15_Center', 'F16_Center']

    labels = master_file[columns_to_transform].apply(lambda row: np.concatenate([row['F1_Center'], row['F2_Center'],
                                                                                 row['F3_Center'], row['F4_Center'],
                                                                                 row['F5_Center'], row['F6_Center'],
                                                                                 row['F7_Center'], row['F8_Center'],
                                                                                 row['F9_Center'], row['F10_Center'],
                                                                                 row['F11_Center'], row['F12_Center'],
                                                                                 row['F13_Center'], row['F14_Center'],
                                                                                 row['F15_Center'], row['F16_Center']])
                                                                 / config['image_size'], axis=1)

    # Create Dataset and DataLoader
    train_dataset = VideoDataset(video_files=video_files, labels=labels, transform=transform)

    # get val master file
    val_master_file_path = os.path.join(config['data_path'], config['master_file_name_val'])
    val_master_file = read_master(config, val_master_file_path)

    # Example Video Files and Labels
    val_video_files = val_master_file['Saved_as_Stack']

    val_labels = val_master_file[columns_to_transform].apply(
        lambda row: np.concatenate([row['F1_Center'], row['F2_Center'], row['F3_Center'], row['F4_Center'],
                                    row['F5_Center'], row['F6_Center'],
                                    row['F7_Center'], row['F8_Center'],
                                    row['F9_Center'], row['F10_Center'],
                                    row['F11_Center'], row['F12_Center'],
                                    row['F13_Center'], row['F14_Center'],
                                    row['F15_Center'], row['F16_Center']]) / config[
                        'image_size'], axis=1)

    # Create Dataset and DataLoader
    val_dataset = VideoDataset(video_files=val_video_files, labels=val_labels, transform=transform)

    print("Training Videos: " + str(train_dataset.__len__()))
    print("Validation Videos: " + str(val_dataset.__len__()))

    num_epochs = config['num_epochs']
    batch_size = config['batch_size']
    model_name = model_ckpt.split("/")[-1]
    new_model_name = f"{model_name}-finetuned-" + str(train_dataset.__len__())
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
        max_steps=(train_dataset.__len__() // batch_size // config['num_gpus']) * num_epochs,
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

else:

    checkpoint_path = os.path.join('videomae-base-finetuned-1017436', 'checkpoint-30000')
    model = VideoMAEForVideoClassification.from_pretrained(checkpoint_path, local_files_only=True)
    image_processor = VideoMAEImageProcessor.from_pretrained(checkpoint_path)
    model.loss_fct = nn.L1Loss()

    mean = []
    if config['mean']:
        mean = config['mean']
    else:
        mean = image_processor.image_mean  # for RGB videos typical values
    std = []
    if config['std']:
        std = config['std']
    else:
        std = image_processor.image_std

        # Data Transformations for both val and train
        transform = transforms.Compose([
            Lambda(lambda x: x / 255.0),
            Normalize(mean, std),

        ])

    # get val master file
    val_master_file_path = os.path.join(config['data_path'], config['master_file_name_val'])
    val_master_file = read_master(config, val_master_file_path)

    asteroid_present = val_master_file[val_master_file['Asteroid_Present'] == True]
    not_present = val_master_file[val_master_file['Asteroid_Present'] == False].reset_index(drop=True)

    # Example Video Files and Labels
    ap_video_files = asteroid_present['Saved_as_Stack']
    ap_labels = asteroid_present['F1_Center'].apply(
        lambda row: (row[0] / config['image_size'], row[1] / config['image_size']))

    # Create Dataset and DataLoader
    ap_dataset = VideoDataset(video_files=ap_video_files, labels=ap_labels, transform=transform)

    # Create a DataLoader
    ap_dataloader = torch.utils.data.DataLoader(
        ap_dataset,
        batch_size=1,  # Adjust batch size as needed
        num_workers=4,  # Adjust number of workers as needed
    )

    # Example Video Files and Labels
    np_video_files = not_present['Saved_as_Stack']
    np_labels = not_present['F1_Center'].apply(
        lambda row: (row[0] / config['image_size'], row[1] / config['image_size']))

    # Create Dataset and DataLoader
    np_dataset = VideoDataset(video_files=np_video_files, labels=np_labels, transform=transform)

    # Create a DataLoader
    np_dataloader = torch.utils.data.DataLoader(
        np_dataset,
        batch_size=1,  # Adjust batch size as needed
        num_workers=4,  # Adjust number of workers as needed
    )

    # Evaluation
    model.eval()
    np_total_mae = 0.0
    with torch.no_grad():
        for batch in np_dataloader:
            videos = batch['video']
            labels = batch['label']
            outputs = model(videos)
            print('\n Not Present')
            predictions = outputs.logits
            print('Predicted Pixel Location: ' + str(config['image_size'] * predictions))
            print('Actual Pixel Location: ' + str(config['image_size'] * labels))

            loss = model.loss_fct(predictions, labels)
            print('MAE: ' + str(config['image_size'] * loss) + '\n')
            np_total_mae += loss.item()

    np_avg_mae = np_total_mae / len(np_dataloader)
    print(f"Average MAE on validation set: {config['image_size'] * np_avg_mae}")

    # Evaluation
    model.eval()
    total_mae = 0.0
    with torch.no_grad():
        for batch in ap_dataloader:
            videos = batch['video']
            labels = batch['label']
            outputs = model(videos)
            print('\n Asteroid Present')
            predictions = outputs.logits
            print('Predicted Pixel Location: ' + str(config['image_size'] * predictions))
            print('Actual Pixel Location: ' + str(config['image_size'] * labels))

            loss = model.loss_fct(predictions, labels)
            print('MAE: ' + str(config['image_size'] * loss) + '\n')
            total_mae += loss.item()

    avg_mae = total_mae / len(ap_dataloader)
    print(f"Average MSE on validation set: {config['image_size'] * avg_mae}")

    # Example Inference
    # video_tensor, _ = dataset[0]  # Get the first video tensor
    # video_tensor = video_tensor.unsqueeze(0)  # Add batch dimension
    #
    # outputs = model(video_tensor)
    # predicted_x, predicted_y = outputs.logits[0].detach().numpy()  # Get predicted (x, y) location
    # print(f"Predicted (x, y): ({predicted_x}, {predicted_y})")
