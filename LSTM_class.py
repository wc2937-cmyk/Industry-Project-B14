# ==========================
#  LSTM MULTI-OUTPUT FITTER
# ==========================
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional

class _LSTMBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim1: int = 64, hidden_dim2: int = 32):
        super().__init__()
        self.l1 = nn.LSTM(input_dim, hidden_dim1, batch_first=True)
        self.l2 = nn.LSTM(hidden_dim1, hidden_dim2, batch_first=True)
        self.fc = nn.Linear(hidden_dim2, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape (batch, 1, feat)
        out, _ = self.l1(x)
        out, _ = self.l2(out)
        out = out[:, -1, :]
        return self.fc(self.relu(out))


class _LSTMWrapper:
    """ 对齐 sklearn 的 predict API """
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def predict(self, X):
        self.model.eval()
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        X = X.unsqueeze(1)               # (N,1,F)
        with torch.no_grad():
            pred = self.model(X).cpu().numpy()
        return pred                      # (N,1)


class LSTMRegressorTrainer:
    """
    与 TriRegressorTrainer 对齐的接口。
    支持在 __init__ 设默认超参，在 fit 里覆盖：
      - lr, max_epochs
      - hidden_dim1, hidden_dim2
      - batch_size
    """
    def __init__(
        self,
        lr: float = 1e-3,
        max_epochs: int = 30,
        hidden_dim1: int = 64,
        hidden_dim2: int = 32,
        batch_size: int = 64,
    ):
        self.models_: List[Tuple[str, object]] = []
        self.lr = lr
        self.max_epochs = max_epochs
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(
        self,
        X,
        y,
        train_ratio: float = 0.8,
        random_state: int = 42,
        shuffle: bool = False,
        lr: Optional[float] = None,
        max_epochs: Optional[int] = None,
        hidden_dim1: Optional[int] = None,
        hidden_dim2: Optional[int] = None,
        batch_size: Optional[int] = None,
    ):
        """
        新增可调参数（传 None 则使用 __init__ 的默认值）：
        - lr
        - max_epochs
        - hidden_dim1, hidden_dim2
        - batch_size
        """
        X = np.asarray(X, float)
        y = np.asarray(y, float)

        N = len(X)
        split = int(N * train_ratio)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        n_outputs = y.shape[1] if y.ndim == 2 else 1
        self.models_ = []

        # 有效超参
        eff_lr = lr if lr is not None else self.lr
        eff_epochs = max_epochs if max_epochs is not None else self.max_epochs
        eff_h1 = hidden_dim1 if hidden_dim1 is not None else self.hidden_dim1
        eff_h2 = hidden_dim2 if hidden_dim2 is not None else self.hidden_dim2
        eff_bs = batch_size if batch_size is not None else self.batch_size
        eff_bs = max(1, int(eff_bs))

        for j in range(n_outputs):

            yj = y_train[:, j] if n_outputs > 1 else y_train
            yj = torch.tensor(yj, dtype=torch.float32).to(self.device)

            model = _LSTMBlock(
                input_dim=X.shape[1],
                hidden_dim1=eff_h1,
                hidden_dim2=eff_h2,
            ).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=eff_lr)
            loss_fn = nn.MSELoss()

            X_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
            X_tensor = X_tensor.unsqueeze(1)    # (N,1,F)

            # === mini-batch train ===
            num_samples = X_tensor.shape[0]
            for ep in range(eff_epochs):
                model.train()
                # 这里简单不用 shuffle，想更随机可以加 permutation
                for start in range(0, num_samples, eff_bs):
                    end = min(start + eff_bs, num_samples)
                    xb = X_tensor[start:end]
                    yb = yj[start:end]

                    optimizer.zero_grad()
                    pred = model(xb).view(-1)
                    loss = loss_fn(pred, yb)
                    loss.backward()
                    optimizer.step()

            wrapped = _LSTMWrapper(model, self.device)
            name = f"LSTM_{j}"
            self.models_.append((name, wrapped))

        return self.models_

