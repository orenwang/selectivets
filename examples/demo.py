import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

from selectivets.config import CfgNode, cfg_fromjson
from selectivets.train import train
from selectivets.test import test
from selectivets.utils import visualize


if __name__ == "__main__":
    cfg = cfg_fromjson('examples/config/SelectiveTimeseries_20200329.json')
    cfg.NUM_EPOCHS = 2  # Only for demo.
    cfg.NN_TYPE = 'LinearSingleFeature'

    model = train(cfg)
    test(cfg, model)
    visualize(cfg, model)

