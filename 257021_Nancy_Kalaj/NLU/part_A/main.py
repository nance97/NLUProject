import argparse
from utils import ensure_atis, prepare_data
from functions import *

# Paths settings
TRAIN_DATA_PATH = "dataset/ATIS/train.json"
TEST_DATA_PATH = "dataset/ATIS/test.json"
TESTING_MODELS_PATH = "bin"
TRAINING_MODELS_PATH = "training_results/models"
LOG_PATH = "training_results/experiments_log.csv"
PLOT_PATH = "training_results/plots"

# Default training hyperparameters
params = {
    "lr": 0.0005,
    "hid_size": 200,
    "emb_size": 300,
    "dropout": 0.3,
    "tr_batch_size": 128,
    "clip": 5,
    "patience_init": 3,
    "n_epochs": 100,
    "runs": 5
}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--exp", required=True)
    p.add_argument("--test", action="store_true")
    args = p.parse_args()

    cfg_mod = __import__(f"configs.{args.exp}", fromlist=['CFG'])
    cfg = cfg_mod.CFG

    ensure_atis()

    # Prepare data
    train_loader, dev_loader, test_loader, lang = prepare_data(
        TRAIN_DATA_PATH, TEST_DATA_PATH
    )
    print(f"First train batch shapes: ")
    first_batch = next(iter(train_loader))
    print("  Utterances:", first_batch['utterances'].shape, "dtype:", first_batch['utterances'].dtype)
    print("  Slot labels:", first_batch['y_slots'].shape, "dtype:", first_batch['y_slots'].dtype)
    print("  Intents:", first_batch['intents'].shape, "dtype:", first_batch['intents'].dtype)
    print("  Device:", first_batch['utterances'].device)


    # Define the loss functions
    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss()

    out_slot = len(lang.slot2id)
    out_int = len(lang.intent2id)
    vocab_len = len(lang.word2id)
    print(f"Vocab size: {vocab_len} | Slots: {out_slot} | Intents: {out_int}")

    # Select the hyperparameters
    select_params(params)
    
    # Train the model
    results = train_model(
        train_loader, dev_loader, test_loader, lang, out_int, out_slot, 
        vocab_len, criterion_slots, criterion_intents, params
    )
