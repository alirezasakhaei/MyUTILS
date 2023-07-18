# Official Library loadings

from datasets import load_dataset
import datasets
import numpy as np
from torch.utils.data import DataLoader
from torch import optim
from torch.nn import CrossEntropyLoss
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import get_peft_model, LoraConfig, TaskType
import os
from tqdm import tqdm

# My loadings
from ARGS import get_ARGS

######################################################
################### BERT FineTuner ###################
######################################################
def calc_acc(model, tokenizer, loader, ARGS):
    model.eval()
    DEVICE = model.device
    t = 0
    c = 0
    for data in loader:
        tokenized = tokenizer(data[ARGS['text_name']], truncation=True, padding=True, return_tensors='pt')
        input_ids = tokenized['input_ids'].to(DEVICE)
        mask = tokenized['attention_mask'].to(DEVICE)
        target = data[ARGS['target_name']].to(DEVICE)
        logits = model(input_ids, mask).logits
        _, selections = torch.max(logits, 1)
        num_corrects = torch.sum(selections == target).item()

        c += num_corrects
        t += target.shape[0]

    return round(c / t, 3)


def Finetune_for_classification(model_name_on_hf, train_dataset, test_dataset, num_labels=2, training_args=None):

    ARGS = get_ARGS()
    verbose = ARGS['verbose']

    if training_args is not None:
        for (key, value) in training_args.items():
            ARGS[key] = value

    tokenizer = AutoTokenizer.from_pretrained(model_name_on_hf)
    model = AutoModelForSequenceClassification.from_pretrained(model_name_on_hf, device_map=ARGS['device_map'],
                                                               num_labels=num_labels)

    if verbose:
        print()
        print()
        print('-' * 30)
        print('Model and Tokenizer loaded successfully!')

    if ARGS['lora']:
        peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, inference_mode=False, r=ARGS['lora_r'],
                                 lora_alpha=ARGS['lora_alpha'], lora_dropout=ARGS['lora_dropout'])
        model = get_peft_model(model, peft_config)

        if verbose:
            print('Lora is applied to the model!')
            model.print_trainable_parameters()

    train_loader = DataLoader(train_dataset, batch_size=ARGS['train_batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=ARGS['test_batch_size'], shuffle=False)

    DEVICE = model.device
    optimizer = optim.AdamW(model.parameters(), lr=ARGS['lr'])
    criteria = CrossEntropyLoss()

    if ARGS['return_accs']:
        train_accs = []
        test_accs = []

    EPOCHS = ARGS['epochs']
    if verbose:
        print('Training ...')
    for epoch in range(EPOCHS):
        if verbose:
            print(f'Epoch: {epoch + 1} / {EPOCHS}')

        model.train()

        loader = train_loader
        if verbose:
            loader = tqdm(train_loader)
        for data in loader:
            tokenized = tokenizer(data[ARGS['text_name']], truncation=True, padding=True, return_tensors='pt')
            input_ids = tokenized['input_ids'].to(DEVICE)
            mask = tokenized['attention_mask'].to(DEVICE)
            target = data[ARGS['target_name']].to(DEVICE)
            logits = model(input_ids, mask).logits
            loss = criteria(logits, target)

            model.zero_grad()
            loss.backward()
            optimizer.step()

        train_acc = calc_acc(model, tokenizer, train_loader, ARGS)
        test_acc = calc_acc(model, tokenizer, test_loader, ARGS)
        if ARGS['return_accs']:
            train_accs.append(train_acc)
            test_accs.append(test_acc)

        if verbose:
            print(f'Train acc is {train_acc} and test acc is {test_acc}')

        if ARGS['save_per_epoch']:
            state_dict = model.state_dict()
            path = ARGS['save_path']

            if not os.path.exists(path):
                os.makedirs(path)
                if verbose:
                    print(f"Folder '{path}' created.")
            else:
                if verbose:
                    print(f"Folder '{path}' already exists.")

            path += os.sep + 'model_checkpoint_epoch' + str(epoch).zfill(ARGS['zfill'])
            torch.save(state_dict, path)
            if verbose:
                print('model saved')
                print('-' * 30)

    RETURNS = {}
    if ARGS['return_model']:
        RETURNS['model'] = model
    if ARGS['return_accs']:
        RETURNS['train_acc'] = train_accs
        RETURNS['test_acc'] = test_accs
    return RETURNS


if __name__ == '__main__':
    model_name = 'bert-base-cased'
    dataset = load_dataset("carblacac/twitter-sentiment-analysis")
    train_dataset = dataset['train']
    test_dataset = dataset['test']
    args = {
        'text_name': 'text',
        'target_name': 'feeling'
    }
    Finetune_for_classification(model_name, train_dataset, test_dataset, num_labels=2, training_args=args)
