import torch


class SelectiveLoss(torch.nn.Module):
    def __init__(self, loss_func, is_selective):
        super(SelectiveLoss, self).__init__()

        self.loss_func = loss_func
        self.is_selective = is_selective

    def forward(self, prediction_out, selection_out, target):
        """
        Args:
            prediction_out: (B,num_classes)
            selection_out:  (B, 1)
        """
        emprical_coverage = selection_out.mean() 
        mse = self.loss_func(prediction_out, target)

        if self.is_selective:
            emprical_risk = (self.loss_func(prediction_out, target)*selection_out).mean()
            emprical_risk = emprical_risk / emprical_coverage
        else:
            emprical_risk = (self.loss_func(prediction_out, target)).mean()

        selective_loss = emprical_risk

        return selective_loss
