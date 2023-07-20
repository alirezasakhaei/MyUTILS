from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd


def infer_num_labels(train_df):
    targets = train_df[-1].values.tolist()
    targets = set(targets)
    num_labels = len(targets)
    return num_labels


def train_model(model_type, model_name_hf, train_df, eval_df, num_labels=None, args=None):
    if not isinstance(train_df, pd.DataFrame):
        train_df = pd.DataFrame(train_df)
    if not isinstance(eval_df, pd.DataFrame):
        eval_df = pd.DataFrame(eval_df)

    if num_labels is None:
        num_labels = infer_num_labels(train_df)

    if args is not None:
        model = ClassificationModel(model_type, model_name_hf, num_labels=num_labels, args=args)
    else:
        model = ClassificationModel(model_type, model_name_hf, num_labels=num_labels)

    print('Model is successfully loaded!')


def get_custom_args(epochs=None, lr=None, early_stopping=False):
    args = ClassificationArgs()
    if epochs is not None:
        args.num_train_epochs = epochs
    if lr is not None:
        args.learning_rate = lr
    if early_stopping:
        args.use_early_stopping = True
        args.early_stopping_delta = 0.01
        args.early_stopping_metric = "mcc"
        args.early_stopping_metric_minimize = False
        args.early_stopping_patience = 5
        args.evaluate_during_training_steps = 1000
    return args



