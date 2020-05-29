import json
from dataclasses import dataclass

@dataclass
class CfgNode:
    COLS = ['Adj Close']

    NUM_EPOCHS: int = 300
    BASE_LR: float = 0.001
    MOMENTUM: float = 0.9
    WEIGHT_DECAY: float = 0.0001
    BATCH_SIZE: int = 128

    INPUT_SIZE: int = 1
    SEQ_LEN: int = 240
    HIDDEN_DIM: int = 512
    BODY_OUT_DIM: int = 1
    LSTM_OUT_DIM: int = 1
    LSTM_NUM_LAYERS: int = 1
    NUM_CLASSES: int = 1
    TRAIN_TEST_SPLIT = [0, 0.9, 1]
    NN_TYPE: str = 'Linear'
    IS_SELECTIVE: bool = True

    ALPHA: float = 0.5
    THRESHOLD: float = 0.5

    VERBOSE: bool = False

    def __post_init__(self):
        self.INPUT_SIZE = len(self.COLS)

def cfg_fromjson(json_fn):
    """Init CfgNode and overwrite the keys given in the json file"""
    cfg = CfgNode()

    jsondata = json.load(open(json_fn, encoding='utf-8'))
    assert 'DATA_PATH' in jsondata, f"Config file '{json_fn}' does not have DATA_PATH!"

    # Overwrite the given keys
    for k, v in jsondata.items():
        setattr(cfg, k, v)

    cfg.__post_init__() # Adjust the computed fields to the updated cfg
    return cfg
