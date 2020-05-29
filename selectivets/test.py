import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data.sampler import SubsetRandomSampler

from selectivets.dataset import TsDataset
from selectivets.model import SelectiveNet, BodyBlock
from selectivets.loss import SelectiveLoss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test(cfg, model):
    """
    Testing.
    """
    model.eval()

    # Data
    dataset = TsDataset(cfg)
    start, split, end = cfg.TRAIN_TEST_SPLIT
    split_idx = int(len(dataset) * split)
    end_idx = int(len(dataset) * end)
    test_subset = Subset(dataset, list(range(split_idx, end_idx)))
    test_loader = DataLoader(test_subset, batch_size=cfg.BATCH_SIZE)

    # Loss used in training
    loss = SelectiveLoss(torch.nn.MSELoss(reduction='none'), cfg.IS_SELECTIVE)

    with torch.no_grad():
        
        test_total_loss = 0.0
        test_mse_loss = 0.0
        test_selected_total_loss = 0.0
        test_selected_mse_loss = 0.0
        selected_count = 0

        for X, y in test_loader:
            x = X.to(device)
            t = y.to(device)

            # forward
            out_class, out_select, out_aux = model(x)

            # compute total loss & plain loss
            for i, _ in enumerate(out_class):
                each_out_class = out_class[i:i+1]
                each_out_select = out_select[i:i+1]
                each_out_aux = out_aux[i:i+1]
                each_t = t[i:i+1]
                selective_loss = loss(each_out_class, each_out_select, each_t)
                selective_loss *= cfg.ALPHA
                aux_loss = (1 - cfg.ALPHA) * torch.nn.MSELoss()(each_out_aux, each_t)
                total_loss = selective_loss + aux_loss
                plain_loss = torch.nn.MSELoss()(each_out_class, each_t)

                # Stats
                test_total_loss += total_loss.item()
                test_mse_loss += plain_loss.item()

                if each_out_select.item() > 0.5:   # Ignore the abstained
                    test_selected_total_loss += total_loss.item()
                    test_selected_mse_loss += plain_loss.item()
                    selected_count += 1
        
        print('='*64)
        print(f'test_total_loss: {test_total_loss / len(test_subset)}')
        print(f'test_mse_loss: {test_mse_loss / len(test_subset)}')
        if selected_count != 0:
            print(f'test_selected_total_loss: {test_selected_total_loss / selected_count}')
            print(f'test_selected_mse_loss: {test_selected_mse_loss / selected_count}')
        print(f'total_count: {len(test_subset)}')
        print(f'selected_count: {selected_count}')
        print('='*64)
