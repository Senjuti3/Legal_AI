import os
import re
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)

# =====================================================================
# CONFIGURATION & API KEY (Set your manual keys here if needed)
# =====================================================================
HF_TOKEN = "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"  # Enter your Hugging Face API key/token here (optional)

if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN

MODEL_NAME = "t5-small"
OUTPUT_DIR = "./saved_legal_model"
TRAIN_FILE = "data/train_data.csv"
VAL_FILE = "data/val_data.csv"

# Hyperparameters
MAX_INPUT_LENGTH = 128
MAX_TARGET_LENGTH = 384
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 5e-4

# Select device (automatically uses GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# =====================================================================
# 1. TEXT CLEANING
# =====================================================================
def clean_data(text):
    text = re.sub(r"\r\n", " ", str(text))
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    return text.strip()

# =====================================================================
# 2. TOKENIZATION
# =====================================================================
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

def tokenize_batch(batch):
    # Clean and tokenize user inputs (problems)
    inputs = [clean_data(text) for text in batch["input_text"]]
    targets = [clean_data(text) for text in batch["target_text"]]

    model_inputs = tokenizer(
        inputs,
        max_length=MAX_INPUT_LENGTH,
        padding="max_length",
        truncation=True
    )

    labels = tokenizer(
        targets,
        max_length=MAX_TARGET_LENGTH,
        padding="max_length",
        truncation=True
    )

    # Replace padding token id's with -100 so they are ignored in loss computation
    labels_matrix = [
        [(label if label != tokenizer.pad_token_id else -100) for label in seq]
        for seq in labels["input_ids"]
    ]

    model_inputs["labels"] = labels_matrix
    return model_inputs

# =====================================================================
# 3. LOAD DATA & TRAIN
# =====================================================================
def main():
    print("Loading datasets...")
    train_df = pd.read_csv(TRAIN_FILE)
    val_df = pd.read_csv(VAL_FILE)

    train_dataset = Dataset.from_pandas(train_df[["input_text", "target_text"]])
    val_dataset = Dataset.from_pandas(val_df[["input_text", "target_text"]])

    print("Tokenizing datasets...")
    train_tokenized = train_dataset.map(tokenize_batch, batched=True)
    val_tokenized = val_dataset.map(tokenize_batch, batched=True)

    print(f"Loading base model '{MODEL_NAME}'...")
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.to(device)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir="./checkpoints",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        save_total_limit=1,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),  # Enabled automatically on GPU
        logging_steps=50,
        report_to="none"
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        processing_class=tokenizer,
        data_collator=data_collator
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving fine-tuned model and tokenizer to '{OUTPUT_DIR}'...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Model training and saving complete!")

if __name__ == "__main__":
    main()
