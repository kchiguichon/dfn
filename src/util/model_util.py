# Copyright 2019 Kenneth Chiguichon
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
