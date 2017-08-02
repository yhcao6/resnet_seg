#!/usr/local/bin/python
# -*- coding: utf-8 -*-

"""docstring
"""

import os
import sys
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

parrots_home = os.environ.get('PARROTS_HOME')
sys.path.append(os.path.join(parrots_home, 'parrots', 'python'))

# from pyparrots.dnn import Session, Model, config
from parrots.dnn import Session, Model, config

def model_and_session(model_file, session_file):
    with open(model_file) as fin:
        model_text = fin.read()
    with open(session_file) as fin:
        session_cfg = yaml.load(fin.read(), Loader=Loader)
    session_cfg = config.ConfigDict(session_cfg)
    session_cfg = config.ConfigDict.to_dict(session_cfg)
    session_text = yaml.dump(session_cfg, Dumper=Dumper)
    print session_text

    model = Model.from_yaml_text(model_text)
    session = Session.from_yaml_text(model, session_text)

    return model, session

