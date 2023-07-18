def get_ARGS():
    ARGS = {'epochs': 3,
            'train_batch_size': 16,
            'test_batch_size': 16,
            'lora': False,
            'lora_r': 8,
            'lora_alpha': 32,
            'lora_dropout': 0.1,
            'lr': 1e-3,
            'verbose': True,
            'device_map': 'cuda',
            'save_per_epoch': True,
            'save_path': '/content/saved_models',
            'text_name': 'question',
            'target_name': 'idx',
            'zfill': 2,
            'return_model': True,
            'return_accs': True
            }
    return ARGS
