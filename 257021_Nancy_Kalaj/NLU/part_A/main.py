from utils import prepare_data
from functions import *

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
    # Select mode and model
    select_config(configs)
    model_filename = f"{get_config(configs)}.pt"
    model_path = os.path.join(TRAINING_MODELS_PATH if configs["training"] else TESTING_MODELS_PATH, model_filename)

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

    if configs["training"]: # Training mode
        # Select the hyperparameters
        select_params(params)
        
        # Train the model
        results = train_model(
            train_loader, dev_loader, test_loader, lang, out_int, out_slot, 
            vocab_len, criterion_slots, criterion_intents, params, configs
        )

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
        print(f"Saved model data as {model_filename}\n")

        # Log and plot results
        log_results(configs, params, results, LOG_PATH)
        plot_results(configs, results, LOG_PATH, PLOT_PATH)

    else: # Testing mode
        # Load the existing model
        ref_model = load_model_data(model_path, out_int, out_slot, vocab_len, configs)

        # Evaluate the existing model performances
        ref_results, ref_intent, _ = eval_loop(test_loader, criterion_slots, criterion_intents, ref_model, lang)

        # Show results
        print("\n==================== Test Results ====================")
        print(f"Results on test set of model with {get_config(configs)}: slot f1 {ref_results['total']['f']}, intent accuracy {ref_intent['accuracy']}")
        print("=====================================================\n")