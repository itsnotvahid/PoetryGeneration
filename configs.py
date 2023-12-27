import torch.cuda

configs = {
        'unfold': 100,
        'step': 1,
        'd_model': 300,
        'num_heads': 6,
        'decoder_layer': 2,
        'dropout': 0.3,
        'lr': 0.1,
        'wd': 1e-6,
        'batch_size': 120,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
           }
