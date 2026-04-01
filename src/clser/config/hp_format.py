

hp_dict: dict = {
    "lr": {
        "hp_lr_min": 1e-4,
        "hp_lr_max": 5e-4
    },
    "per_device_train_batch_size": [4, 8, 16],
    "warmup_ratio": {
        "warmup_ratio_min": 0.0,
        "warmup_ratio_max": 0.1
    }
}

def hp_space(trial):

    global hp_dict

    return_dict = {}
    if "lr" in hp_dict:
        return_dict["learning_rate"] = trial.suggest_float(
            "learning_rate", float(hp_dict["lr"]["hp_lr_min"]), float(hp_dict["lr"]["hp_lr_max"]), log=True
            )
    if "per_device_train_batch_size" in hp_dict:
        assert isinstance(hp_dict["per_device_train_batch_size"], list), "Batch size just support specified num."
        return_dict["per_device_train_batch_size"] = trial.suggest_categorical(
                "per_device_train_batch_size", hp_dict["per_device_train_batch_size"]
            )
    if "warmup_ratio" in hp_dict:
        return_dict["warmup_ratio"] = trial.suggest_float(
                "warmup_ratio", float(hp_dict["warmup_ratio"]["warmup_ratio_min"]), float(hp_dict["warmup_ratio"]["warmup_ratio_max"])
            )
        
    if "weight_decay" in hp_dict:
        return_dict["weight_decay"] = trial.suggest_float(
                "weight_decay", float(hp_dict["weight_decay"]["weight_decay_min"]), float(hp_dict["weight_decay"]["weight_decay_max"])
            )
    
    return return_dict
