from datasets import load_dataset
import datasets
import numpy as np
######################################################
########### Dataset Fixer For CLassification ##########
######################################################

# def fit_mcq_into_template(dataset = datasets.dataset_dict.DatasetDict()):
#     dataset = dataset.rename_column('answer', 'target')
#
#     new_column = ["foo"] * len(dataset)
#     dataset = dataset.add_column("new_column", new_column)
#     temp = np.zeros(len(dataset))
#     dataset['optA'] = temp
#
#     return dataset
#
#
#
# dataset = load_dataset('tasksource/mmlu', 'abstract_algebra')
# dataset = fit_mcq_into_template(dataset)
# print(type(dataset))
# print(dataset['test'])