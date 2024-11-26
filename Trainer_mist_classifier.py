import numpy as np
import torch
from docutils.nodes import header
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
import pathlib

print("Loaded pytorchvideo")
from torchvision.transforms import (
    Lambda
)
from sklearn.metrics import confusion_matrix, classification_report

# metric = evaluate.load(os.path.join(os.environ.get('SCRATCH'), 'huggingface', 'metrics', 'accuracy', 'default', 'accuracy.py'))
metric = evaluate.load("accuracy")

# Define a converter function
def str_to_tuple(x):
    try:
        return ast.literal_eval(x)
    except (ValueError, SyntaxError):
        return x


def create_test_file(config):
    # Directory containing the two subdirectories
    root_directory = config['test_set_path']

    # Initialize list to store file paths and labels
    data = []

    # Traverse the directory
    for label_dir in config['labels']:
        label = True if label_dir == config['labels'][0] else False
        folder_path = os.path.join(root_directory, label_dir)

        # Ensure the folder exists to avoid errors
        if os.path.exists(folder_path):
            for file_name in os.listdir(folder_path):
                if file_name.endswith(".npy"):  # Only process .npy files
                    file_path = os.path.join(folder_path, file_name)
                    data.append({"file_path": file_path, "label": label})

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Save to CSV
    df.to_csv(os.path.join(config['test_set_path'], config['master_file_name_test']), index=False)
    return

def read_master(configuration, master_file_path):
    columns_to_convert = configuration['center_file_columns']
    columns_to_convert_final = ['Stack_Crop_Start'] + columns_to_convert
    return pd.read_csv(master_file_path, sep=',', header=0,
                       converters={col: str_to_tuple for col in columns_to_convert_final},
                       names=configuration['master_file_columns'])


def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)


def collate_fn(examples):
    pixel_values = torch.stack(
        [example['video'] for example in examples]
    ).to(torch.float32)
    labels = torch.stack([example['label'] for example in examples])
    output = {"pixel_values": pixel_values, "labels": labels}
    return output


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
        label = torch.tensor(self.labels[idx])  # do not specify type for some reason

        # Load the video and extract frames
        video_array = np.load(video_path)
        video_tensor = torch.from_numpy(video_array)
        name = video_path.split('/')[-1]

        if self.transform:
            video_tensor = self.transform(video_tensor.permute(1,0,2,3))
        video_tensor = video_tensor.permute(1,0,2,3)

        return {"name": name,"video": video_tensor, "label": label}


with open('config_classifier.yaml', 'r') as f:
    config = yaml.safe_load(f)

dataset_root_path = config['data_path'] + 'vids/'
dataset_root_path = pathlib.Path(dataset_root_path)

all_video_file_paths = (
        list(dataset_root_path.glob("train/*/*.npy"))
        + list(dataset_root_path.glob("val/*/*.npy"))
)

class_labels = sorted({str(path).split("/")[-2] for path in all_video_file_paths})
label2id = {class_labels[0]: int(0), class_labels[1]: int(1)}
# label2id = {label: i for i, label in enumerate(class_labels)}
id2label = {i: label for label, i in label2id.items()}
print(id2label)


print(f"Unique classes: {list(label2id.keys())}.")

if config['train']:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Device: " + str(device))

    # MCG-NJU/videomae-base-finetuned-kinetics.
    model_ckpt = config['model_name']

    # Load the VideoMAE model for regression
    image_processor = VideoMAEImageProcessor.from_pretrained(model_ckpt)
    model = VideoMAEForVideoClassification.from_pretrained(model_ckpt, label2id=label2id, id2label=id2label,
                                                           ignore_mismatched_sizes=True)
    mean = []
    if config['mean']:
        mean = config['mean']
    else:
        mean = image_processor.image_mean # for RGB videos typical values
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
    labels = master_file['Asteroid_Present'].map({True: label2id[class_labels[0]], False: label2id[class_labels[1]]})


    # Create Dataset and DataLoader
    train_dataset = VideoDataset(video_files=video_files, labels=labels, transform=transform)

    # get val master file
    val_master_file_path = os.path.join(config['data_path'], config['master_file_name_val'])
    val_master_file = read_master(config, val_master_file_path)

    # Example Video Files and Labels
    val_video_files = val_master_file['Saved_as_Stack']
    val_labels = val_master_file['Asteroid_Present'].map({True: label2id[class_labels[0]], False: label2id[class_labels[1]]})

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
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Device: " + str(device))

    # MCG-NJU/videomae-base-finetuned-kinetics.
    model_ckpt = config['model_name']

    # Load the VideoMAE model for regression
    image_processor = VideoMAEImageProcessor.from_pretrained(model_ckpt)
    model = VideoMAEForVideoClassification.from_pretrained(model_ckpt, label2id=label2id, id2label=id2label,
                                                           ignore_mismatched_sizes=True)
    model.eval()
    model.to(device)

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

    #  get test master file
    test_file_path = os.path.join(config['test_set_path'], config['master_file_name_test'])
    if 'c' in config['options']:
        test_file = read_master(config, test_file_path)
        test_video_files = test_file['Saved_as_Stack']
        test_labels = test_file['Asteroid_Present'].map(
            {True: label2id[class_labels[0]], False: label2id[class_labels[1]]})
        snr_df = pd.DataFrame(columns=['name', 'SNR'])
        snr_df['SNR'] = test_file['Expected_SNR']
        snr_df['name'] = test_file['Saved_as_Stack'].apply(lambda x: x.split('/')[-1])
    else:
        if not os.path.exists(test_file_path):
            create_test_file(config)
        test_file = pd.read_csv(test_file_path, sep=',', header=0, names=config['test_file_columns'])

        # Example Video Files and Labels
        test_video_files = test_file['file_path']
        test_labels = test_file['label'].map({True: label2id[class_labels[0]], False: label2id[class_labels[1]]})

        snr_df = pd.read_csv('snr_values.csv', sep=',', header=0, names=['name', 'SNR'])

    # Create Dataset and DataLoader
    test_dataset = VideoDataset(video_files=test_video_files, labels=test_labels, transform=transform)

    # Create a DataLoader
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,  # Adjust batch size as needed
        num_workers=4,  # Adjust number of workers as needed
    )

    # Inference and results storage
    results = []
    all_preds = []
    all_labels = []

    for batch in test_loader:
        video_tensor = batch["video"].to(device)  # Move video to device
        true_label = batch["label"].item()  # Extract true label
        name = batch["name"][0]
        print(name)

        # Run inference
        with torch.no_grad():
            outputs = model(video_tensor)
            logits = outputs.logits
            pred_label = torch.argmax(logits, dim=1).item()

        # Determine result type (TP, TN, FP, FN)
        if pred_label == true_label:
            if true_label == 0:
                result = "TP"
            else:
                result = "TN"
        else:
            if pred_label == 0:
                result = "FP"
            else:
                result = "FN"


        # Find the SNR value for the sample
        matching_snr = snr_df[snr_df["name"].apply(lambda x: x in name)]
        snr_value = matching_snr["SNR"].values[0] if not matching_snr.empty else None

        if 'c' in config['options']:
            matching = test_file[test_file['Saved_as_Stack'].apply(lambda x: name in x)]
            v_value = matching['H'].values[0]
            om_value = matching['omega'].values[0]

            # Append results
            results.append({
                "sample_name": name,
                "predicted": pred_label,
                "true": true_label,
                "result": result,
                "SNR": snr_value,
                "V": v_value,
                "Omega": om_value
            })

        else:
            # Append results
            results.append({
                "sample_name": name,
                "predicted": pred_label,
                "true": true_label,
                "result": result,
                "SNR": snr_value
            })

        all_preds.append(pred_label)
        all_labels.append(true_label)

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=["Asteroid Present", "Not"])

    print("Confusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(report)

    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(config['output_file_name'], index=False)
    print("Results saved to inference_results.csv")
