# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import json
from . import lstm_models as lm


def create_net(nb_labers=5, labels=None, rnn_type='lstm', bidirectional=False, datapath=None, hidden_size=800, hidden_layers=5, sample_rate=16000, window_size=0.02, window_stride=0.01, window='hamming', noise_dir=None, noise_prob=0.4, noise_min=0.0, noise_max=0.5):
    if labels is None:
        with open(os.path.join(datapath, 'labels.json')) as label_file:
            labels = str(''.join(json.load(label_file)))

    print(" =============  audio_conf preparing =================, datapat: ", datapath)
    audio_conf = dict(sample_rate=sample_rate,
            window_size=window_size,
            window_stride=window_stride,
            window=window,
            noise_dir=noise_dir,
            noise_prob=noise_prob,
            noise_levels=(noise_min, noise_max))
    
    print(" =============  net preparing =================")
    net = lm.DeepSpeech(rnn_hidden_size=hidden_size,
            nb_layers=hidden_layers,
            labels=labels,
            rnn_type=lm.supported_rnns[rnn_type],
            audio_conf=audio_conf,
            bidirectional=bidirectional)
    ext = {'audio_conf': audio_conf,
            'labels': labels}
    return net, ext 
