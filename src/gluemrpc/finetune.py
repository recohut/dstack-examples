import torch
from torch.utils.data import DataLoader

from accelerate import Accelerator

import datasets
from datasets import load_dataset, load_metric

import transformers
from transformers import (
    AdamW,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
)

from tqdm.auto import tqdm
import shutil
import argparse
from pathlib import Path


def main():
    
    # Finetune settings
    parser = argparse.ArgumentParser(description='Huggingface Accelerate GLUe MRPC Example')
    parser.add_argument('--learning-rate', type=int, default=2e-5, metavar='N',
                        help='learning rate of the fine-tuning steps (default: 2e-5)')
    parser.add_argument('--train-batch-size', type=int, default=8, metavar='N',
                        help='actual train batch size will be x 8 (default: 8)')
    parser.add_argument('--eval-batch-size', type=int, default=32, metavar='N',
                        help='actual eval batch size will be x 8 (default: 32)')
    parser.add_argument('--num-epochs', type=int, default=3, metavar='N',
                        help='number of epochs to fine-tune for (default: 3)')
    parser.add_argument('--fseed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='for Saving the current Model')
    args = parser.parse_args()

    # raw_datasets = load_dataset(path="glue", name="mrpc", cache_dir="data")
    shutil.copytree("data", "hf_dataset")
    raw_datasets = datasets.load_from_disk("hf_dataset")

    model_checkpoint = "bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    def tokenize_function(examples):
        outputs = tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, padding="max_length", max_length=128)
        return outputs

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True, remove_columns=["idx", "sentence1", "sentence2"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")
    
    def create_dataloaders(train_batch_size=8, eval_batch_size=32):
        train_dataloader = DataLoader(
            tokenized_datasets["train"], shuffle=True, batch_size=train_batch_size
        )
        eval_dataloader = DataLoader(
            tokenized_datasets["validation"], shuffle=False, batch_size=eval_batch_size
        )
        return train_dataloader, eval_dataloader

    metric = load_metric("glue", "mrpc")

    def training_function():
        # Initialize accelerator
        accelerator = Accelerator()

        # To have only one message (and not 8) per logs of Transformers or Datasets, we set the logging verbosity
        # to INFO for the main process only.
        if accelerator.is_main_process:
            datasets.utils.logging.set_verbosity_warning()
            transformers.utils.logging.set_verbosity_info()
        else:
            datasets.utils.logging.set_verbosity_error()
            transformers.utils.logging.set_verbosity_error()

        train_dataloader, eval_dataloader = create_dataloaders(
            train_batch_size=args.train_batch_size, eval_batch_size=args.eval_batch_size
        )
        # The seed need to be set before we instantiate the model, as it will determine the random head.
        set_seed(args.fseed)

        # Instantiate the model, let Accelerate handle the device placement.
        model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)

        # Instantiate optimizer
        optimizer = AdamW(params=model.parameters(), lr=args.learning_rate)

        # Prepare everything
        # There is no specific order to remember, we just need to unpack the objects in the same order we gave them to the
        # prepare method.
        model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader
        )

        num_epochs = args.num_epochs
        # Instantiate learning rate scheduler after preparing the training dataloader as the prepare method
        # may change its length.
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=100,
            num_training_steps=len(train_dataloader) * num_epochs,
        )

        # Instantiate a progress bar to keep track of training. Note that we only enable it on the main
        # process to avoid having 8 progress bars.
        progress_bar = tqdm(range(num_epochs * len(train_dataloader)), disable=not accelerator.is_main_process)
        # Now we train the model
        for epoch in range(num_epochs):
            model.train()
            for step, batch in enumerate(train_dataloader):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

            model.eval()
            all_predictions = []
            all_labels = []

            for step, batch in enumerate(eval_dataloader):
                with torch.no_grad():
                    outputs = model(**batch)
                predictions = outputs.logits.argmax(dim=-1)

                # We gather predictions and labels from the 8 TPUs to have them all.
                all_predictions.append(accelerator.gather(predictions))
                all_labels.append(accelerator.gather(batch["labels"]))

            # Concatenate all predictions and labels.
            # The last thing we need to do is to truncate the predictions and labels we concatenated
            # together as the prepared evaluation dataloader has a little bit more elements to make
            # batches of the same size on each process.
            all_predictions = torch.cat(all_predictions)[:len(tokenized_datasets["validation"])]
            all_labels = torch.cat(all_labels)[:len(tokenized_datasets["validation"])]

            eval_metric = metric.compute(predictions=all_predictions, references=all_labels)

            # Use accelerator.print to print only on the main process.
            accelerator.print(f"epoch {epoch}:", eval_metric)
            
            if args.save_model:
                Path("model").mkdir(exist_ok=True)
                accelerator.save_state("model/gluemrpc_epochs={}_lr={}.pt".format(args.num_epochs, args.learning_rate))
            
    training_function()


if __name__ == '__main__':
    main()