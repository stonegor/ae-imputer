class EarlyStopper:
    """
    A class for stopping training early if the loss doesn't decrease significantly.

    Attributes:
        patience (int): How many times to allow the loss to not improve before stopping.
        min_delta (float): The smallest decrease in loss considered as an improvement.

    Methods:
        early_stop(loss): Checks if training should stop based on the current loss.

    Example:
        early_stopper = EarlyStopper(patience=5, min_delta=0.01)
        for epoch in range(max_epochs):
            loss = train_epoch()
            if early_stopper.early_stop(loss):
                break
    """
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self._counter = 0
        self._min_loss = float('inf')

    def early_stop(self, loss):
        if loss < self._min_loss:
            self._min_loss = loss
            self._counter = 0
        elif loss > (self._min_loss - self.min_delta):
            self._counter += 1
            if self._counter >= self.patience:
                return True
        return False