# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
InferSent models. See https://github.com/facebookresearch/InferSent.
"""

from __future__ import absolute_import, division, unicode_literals

import sys
import os
import json
import torch
import logging
import torch
from torch import nn

from src.SentEval import senteval

from transformers import BertTokenizer  # , BertModel
from models.modeling_bert import BertModel

# Set PATHs
PATH_TO_DATA = 'src/SentEval/data'


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')

    def forward(self, batch):
        with torch.no_grad():
            batch = [' '.join(sent) for sent in batch]
            inputs = self.tokenizer(
                batch,
                padding='max_length',
                max_length=128,
                truncation=True,
                return_tensors='pt',
            )

            for k, v in inputs.items():
                inputs[k] = v.cuda()
            
            hidden_states = self.model(**inputs, output_hidden_states=True).hidden_states  # TODO: specify the output
            first_hidden_states, last_hidden_states = hidden_states[0], hidden_states[-1]
            outputs = torch.sum(inputs['attention_mask'].unsqueeze(dim=2) * (first_hidden_states + last_hidden_states) / 2, dim=1)

        return outputs.cpu().numpy()


def batcher(params, batch):
    return params.encoder(batch)


params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                 'tenacity': 3, 'epoch_size': 2}
# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.INFO)


if __name__ == '__main__':
    model = Encoder()
    model.eval()
    model.cuda()
    params_senteval['encoder'] = model

    se = senteval.engine.SE(params_senteval, batcher)
    transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']
    results = se.eval(transfer_tasks)

    for key in results.keys():
        print("Evaluation Result of {}".format(key))
        print(results[key]['all'])

