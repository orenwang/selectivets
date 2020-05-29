import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data.sampler import SubsetRandomSampler
import plotly.graph_objects as go

from selectivets.dataset import TsDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def visualize(cfg, model, mode='test', multiplier_mse=20):
    """
    Plot a given model on test data.
    """
    model.eval()

    # Data
    dataset = TsDataset(cfg)
    start, split, end = cfg.TRAIN_TEST_SPLIT
    start_idx = int(len(dataset) * start)
    split_idx = int(len(dataset) * split)
    end_idx = int(len(dataset) * end)
    if mode == 'test':
        test_subset = Subset(dataset, list(range(split_idx, end_idx)))
    elif mode == 'train':
        test_subset = Subset(dataset, list(range(start_idx, split_idx)))
    test_loader = DataLoader(test_subset, batch_size=cfg.BATCH_SIZE)

    with torch.no_grad():
        
        fig_y = []
        fig_pred = []
        fig_accept_pred = []
        fig_abstain_pred = []
        fig_g = []
        fig_mse = []

        for X, y in test_loader:
            x = X.to(device)
            t = y.to(device)

            # forward
            out_class, out_select, out_aux = model(x)

            fig_y += y.flatten().tolist()
            fig_g += out_select.flatten().tolist()

            # Iterate over this batch
            for i, _ in enumerate(out_class):
                fig_pred.append(out_class[i].item())
                if out_select[i] > 0.5:
                    fig_accept_pred.append(out_class[i].item())
                    fig_abstain_pred.append(None)
                else:
                    fig_accept_pred.append(None)
                    fig_abstain_pred.append(out_class[i].item())
        
        fig_mse = (multiplier_mse * torch.nn.MSELoss(reduction='none')(torch.tensor(fig_y), torch.tensor(fig_pred))).tolist()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(len(fig_y))), y=fig_y, name="fig_y"))
        fig.add_trace(go.Scatter(x=list(range(len(fig_y))), y=fig_pred, name="fig_pred"))
        fig.add_trace(go.Scatter(x=list(range(len(fig_y))), y=fig_abstain_pred, name="fig_abstain_pred"))
        fig.add_trace(go.Scatter(x=list(range(len(fig_y))), y=fig_g, name="fig_g"))
        fig.add_trace(go.Scatter(x=list(range(len(fig_y))), y=fig_mse, name="fig_mse"))
        fig.show()
