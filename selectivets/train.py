import torch
from torch.utils.data import DataLoader, Subset

from selectivets.dataset import TsDataset
from selectivets.model import SelectiveNet, BodyBlock
from selectivets.loss import SelectiveLoss

torch.manual_seed(1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(cfg):
    """
    Training loop.
    """
    # Data
    dataset = TsDataset(cfg)
    start, split, end = cfg.TRAIN_TEST_SPLIT    # e.g. 0, 0.9, 1
    start_idx = int(len(dataset) * start)
    split_idx = int(len(dataset) * split)
    train_subset = Subset(dataset, list(range(start_idx, split_idx)))
    train_loader = DataLoader(train_subset, batch_size=cfg.BATCH_SIZE, shuffle=True)

    # Model & Loss & Optimizer
    body_block = BodyBlock(cfg)
    model = SelectiveNet(cfg, body_block).to(device)
    loss = SelectiveLoss(torch.nn.MSELoss(reduction='none'), cfg.IS_SELECTIVE)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.BASE_LR, momentum=cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY)

    for epoch in range(cfg.NUM_EPOCHS):
        
        # Running stats
        running_loss = 0.0
        running_select = 0.0
        running_cover = 0.0
        count = 0

        for X, y in train_loader:
            x = X.to(device)
            t = y.to(device)

            # forward
            out_class, out_select, out_aux = model(x)

            # compute selective loss & total loss
            selective_loss = loss(out_class, out_select, t)
            selective_loss *= cfg.ALPHA
            aux_loss = (1 - cfg.ALPHA) * torch.nn.MSELoss()(out_aux, t)
            total_loss = selective_loss + aux_loss

            # backward
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Running Stats
            running_loss += total_loss.item()
            running_select += out_select.mean().detach()
            running_cover += (out_select > cfg.THRESHOLD).float().mean().detach()

            count += 1
        
        if cfg.VERBOSE:
            print('[%d] loss: %.7f | select: %.5f | coverage: %.5f' % (epoch + 1, running_loss / count, running_select / count, running_cover / count))

    return model
