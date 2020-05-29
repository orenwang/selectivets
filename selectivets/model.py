import torch


class SelectiveNet(torch.nn.Module):
    """
    SelectiveNet for regression and classification, finding intrinsic coverage with rejection option.
    """
    def __init__(self, cfg, features):
        """
        Args
            features: feature extractor network (called body block in the paper).
            dim_featues: dimension of feature from body block.  
            num_classes: number of classification class.
        """
        super(SelectiveNet, self).__init__()
        self.features = features
        self.input_size = cfg.INPUT_SIZE
        self.hidden_dim = cfg.HIDDEN_DIM
        self.num_classes = cfg.NUM_CLASSES
        
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim, self.num_classes)
        )

        self.selector_1 = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim, 20),
            torch.nn.ReLU(True)
        )
        self.selector_2 = torch.nn.Sequential(
            torch.nn.Linear(20, 1),
            torch.nn.Sigmoid()
        )

        self.selector = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim, 256),
            # torch.nn.ReLU(),
            torch.nn.BatchNorm1d(256),
            torch.nn.Linear(256, 128),
            # torch.nn.ReLU(),
            torch.nn.BatchNorm1d(128, track_running_stats=False),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(128),
            torch.nn.Linear(128, 1),
            torch.nn.Sigmoid()
        )

        self.aux_predictor = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim, self.num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        
        prediction_out = self.predictor(x)
        selection_out  = self.selector(x)
        auxiliary_out  = self.aux_predictor(x)

        return prediction_out, selection_out, auxiliary_out


class BodyBlock(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.nn_type = cfg.NN_TYPE
        self.input_size = cfg.INPUT_SIZE
        self.hidden_dim = cfg.HIDDEN_DIM
        self.out_features = cfg.BODY_OUT_DIM
        self.seq_len = cfg.SEQ_LEN
        self.lstm_out_dim = cfg.LSTM_OUT_DIM
        self.n_layers = cfg.LSTM_NUM_LAYERS
        
        # self.linear = torch.nn.Linear(in_features=input_size, out_features=1)
        # self.linear_hidden2out = torch.nn.Linear(seq_len * 1, out_dim)
        if self.nn_type == 'LinearSingleFeature':
            self.linear = torch.nn.Linear(in_features=self.seq_len, out_features=self.hidden_dim)
        elif self.nn_type == 'Linear':
            self.linear_1 = torch.nn.Linear(in_features=self.input_size, out_features=self.out_features)
            self.linear_2 = torch.nn.Linear(in_features=self.seq_len * self.out_features, out_features=self.hidden_dim)
        elif self.nn_type == 'LSTM':
            self.lstm = torch.nn.LSTM(input_size=self.input_size, hidden_size=self.lstm_out_dim, num_layers=self.n_layers, batch_first=True)
            self.hidden2out = torch.nn.Linear(self.seq_len * self.lstm_out_dim, self.hidden_dim)
            self.lasthidden2out = torch.nn.Linear(self.lstm_out_dim * self.n_layers, self.hidden_dim)
        
    def forward(self, window_of_prices):
        """
        Args:
            window_of_prices (tensor)

        Returns:
            Flattened output.
        """
        if self.nn_type == 'LinearSingleFeature':
            x = window_of_prices.view(-1, self.seq_len)
            out = self.linear(x)
        elif self.nn_type == 'Linear':
            # hidden = self.linear(window_of_prices)
            # out = self.linear_hidden2out(hidden.view(hidden.shape[0], -1))
            x = self.linear_1(window_of_prices)
            x = x.view(-1, self.seq_len * self.out_features)
            out = self.linear_2(x)
        elif self.nn_type == 'LSTM':
            # x = window_of_prices.transpose(0,1)
            lstm_out, hidden = self.lstm(window_of_prices)
            # lstm_out = lstm_out.transpose(0, 1)
            # print(lstm_out.shape)
            # print(self.seq_len * self.lstm_out_dim)
            lstm_out = lstm_out.view(-1, self.seq_len * self.lstm_out_dim)  # Use reshape here because non-contiguous
            # lstm_out = lstm_out.reshape(-1, self.seq_len * self.lstm_out_dim)  # Use reshape here because non-contiguous
            last_hidden_state = hidden[0]
            last_hidden_state = last_hidden_state.transpose(0, 1)
            last_hidden_state = last_hidden_state.reshape(-1, self.lstm_out_dim * self.n_layers)  # Use reshape here because non-contiguous
            
            out = self.hidden2out(lstm_out)
            # out = self.lasthidden2out(last_hidden_state)
            # out = last_hidden_state
        return out
                