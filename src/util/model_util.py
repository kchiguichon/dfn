import os
import json
from data import swap_key_values

def save_model(model, config, vocab, serialization_dir):
    config['model_type'] = model.__class__.__name__
    json.dump(config, open(os.path.join(serialization_dir, f'config.json'), 'w', encoding='utf8'))
    json.dump(vocab, open(os.path.join(serialization_dir, f'vocab.json'), 'w', encoding='utf8'))
    model.save_weights(os.path.join(serialization_dir, f'model.ckpt'))

def load_model(serialization_dir):
    config = json.load(open(os.path.join(serialization_dir, f'config.json'), 'r', encoding='utf8'))
    vocab = json.load(open(os.path.join(serialization_dir, f'vocab.json'), 'r', encoding='utf8'))
    reverse_vocab = swap_key_values(vocab)
    model_type = config['model_type']
    del config['model_type']
    import models.neural_models as neural_models
    model = getattr(neural_models, model_type)(**config)
    model.load_weights(os.path.join(serialization_dir, f'model.ckpt'))
    return model, config, vocab, reverse_vocab