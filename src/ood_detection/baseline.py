from datasets.zoc_loader import IsolatedClasses


def get_trained_linear_classifier(train_set, val_set, seen_labels, clip_model=None, device=None):
    class_to_idx_mapping = train_set.class_to_idx
    train_set_loaders = IsolatedClasses(train_set)
    val_set_loaders = IsolatedClasses(val_set)

    feature_weight_dict_train = get_feature_weight_dict(train_set_loaders, clip_model, device)
    feature_weight_dict_val = get_feature_weight_dict(val_set_loaders, clip_model, device)
    linear_train_set = FeatureSet(feature_weight_dict_train, seen_labels, class_to_idx_mapping)
    linea_val_set = FeatureSet(feature_weight_dict_val, seen_labels, class_to_idx_mapping)

    linear_classifier = train_id_classifier(linear_train_set, linea_val_set, epochs=20, learning_rate=0.001,
                                            wandb_logging=False)
    return linear_classifier.eval()