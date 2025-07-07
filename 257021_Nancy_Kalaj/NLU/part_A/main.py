from utils import prepare_data
from functions import *

import os
import torch
import torch.nn as nn

# Paths settings
TRAIN_DATA_PATH = "dataset/train.json"
TEST_DATA_PATH = "dataset/test.json"
TESTING_MODELS_PATH = "bin"
TRAINING_MODELS_PATH = "training_results/models"
LOG_PATH = "training_results/experiments_log.csv"
PLOT_PATH = "training_results/plots"

# Default configuration settings
configs = {
    "training": True,
    "use_bidir": False,
    "use_dropout": False,
}

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
    print("==== Starting experiment ====")

    # Select mode and model
    select_config(configs)
    print(f"Configs: {configs}")

    model_filename = f"{get_config(configs)}.pt"
    model_path = os.path.join(TRAINING_MODELS_PATH if configs["training"] else TESTING_MODELS_PATH, model_filename)
    print(f"Model will be saved/loaded at: {model_path}")

    # Prepare data
    print("\n---- Preparing data ----")
    train_loader, dev_loader, test_loader, lang, out_slot, out_int, vocab_len = prepare_data(
        TRAIN_DATA_PATH, TEST_DATA_PATH, model_path, params, configs
    )
    print(f"Vocab size: {vocab_len} | Slots: {out_slot} | Intents: {out_int}")
    print(f"First train batch shapes: ")
    first_batch = next(iter(train_loader))
    print("  Utterances:", first_batch['utterances'].shape, "dtype:", first_batch['utterances'].dtype)
    print("  Slot labels:", first_batch['y_slots'].shape, "dtype:", first_batch['y_slots'].dtype)
    print("  Intents:", first_batch['intents'].shape, "dtype:", first_batch['intents'].dtype)
    print("  Device:", first_batch['utterances'].device)

    # Define the loss functions
    print("\n---- Setting loss functions ----")
    print("PAD_TOKEN used:", PAD_TOKEN)
    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss()

    if configs["training"]: # Training mode
        print("\n---- Training Mode ----")
        select_params(params)
        print(f"Training params: {params}")

        print("\n---- Training model ----")
        results = train_model(
            train_loader, dev_loader, test_loader, lang, out_int, out_slot, 
            vocab_len, criterion_slots, criterion_intents, params, configs
        )
        print("Train results (partial):")
        for k in ['slot_f1', 'int_acc', 'losses_dev', 'losses_train', 'sampled_epochs']:
            print(f"  {k}: {results[k] if k in results else 'n/a'}")
        print(f"Best model: {type(results['best_model'])}")

        # Save the model
        os.makedirs(TRAINING_MODELS_PATH, exist_ok=True)
        model_data = {
            "model_state_dict": results["best_model"].state_dict(),
            "params": params,
            "word2id": lang.word2id,
            "slot2id": lang.slot2id,
            "intent2id": lang.intent2id
        }
        torch.save(model_data, model_path)
        print(f"\nModel data saved as {model_filename} at {model_path}")

        # Log and plot results
        print("---- Logging results ----")
        log_results(configs, params, results, LOG_PATH)
        print(f"Results logged at {LOG_PATH}")

        print("---- Plotting results ----")
        plot_results(configs, results, LOG_PATH, PLOT_PATH)
        print(f"Plots saved to {PLOT_PATH}")

    else: # Testing mode
        print("\n---- Testing Mode ----")
        # Load the existing model
        ref_model = load_model_data(model_path, out_int, out_slot, vocab_len, configs)
        print("Loaded model from", model_path)

        # Evaluate the existing model performances
        ref_results, ref_intent, _ = eval_loop(test_loader, criterion_slots, criterion_intents, ref_model, lang)
        print("Evaluated model on test set.")

        # Show results
        print("\n==================== Test Results ====================")
        print(f"Results on test set of model with {get_config(configs)}: slot f1 {ref_results['total']['f']}, intent accuracy {ref_intent['accuracy']}")
        print("=====================================================\n")

    print("==== Done! ====")
