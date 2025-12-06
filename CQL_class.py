# ==========================
#     CQL POLICY TRAINER
# ==========================
import numpy as np
import torch
import d3rlpy
from d3rlpy.dataset import MDPDataset
from typing import Optional, Dict, Any, List, Tuple

class _CQLWrapper:
    def __init__(self, cql_model):
        self.model = cql_model
        self.id2act = {0: -1, 1: 0, 2: 1}

    def predict(self, X):
        X = np.asarray(X, np.float32)
        pred_id = self.model.predict(X)
        return np.array([[self.id2act[i]] for i in pred_id], float)


class CQLPolicyTrainer:
    def __init__(
        self,
        train_ratio: float = 0.8,
        n_steps: int = 3000,
        n_steps_per_epoch: int = 1000,
        batch_size: int = 256,
        cql_config_kwargs: Optional[Dict[str, Any]] = None,
        device: str = "cpu",
    ):
        """
        cql_config_kwargs: 传给 DiscreteCQLConfig 的额外参数，例如
            {"gamma": 0.99, "batch_size": 256}
        n_steps, n_steps_per_epoch: 训练步数相关
        batch_size: 每个训练 step 的 batch size
        """
        self.models_: List[Tuple[str, Any]] = []
        self.train_ratio = train_ratio
        self.n_steps = n_steps
        self.n_steps_per_epoch = n_steps_per_epoch
        self.batch_size = batch_size
        self.cql_config_kwargs = cql_config_kwargs or {}
        self.device = device

    def _build_dataset(self, X: np.ndarray, y: np.ndarray):
        """
        多资产版本：
        - X: (N, F)
        - y: (N,) 或 (N, M)
        """

        X = np.asarray(X, float)
        y = np.asarray(y, float)

        # ===== reward 合成 =====
        if y.ndim == 2:
            reward = y.mean(axis=1)
        else:
            reward = y
        reward = reward.reshape(-1)

        # ===== 对齐 =====
        obs = X[:-1]
        next_obs = X[1:]
        r = reward[:-1]

        # ===== action: sign(r) =====
        sign = np.sign(r)
        action = (sign + 1).astype(int)  # -1,0,1 → 0,1,2

        # ===== terminals: 最后一个 step 为 True =====
        terminals = np.zeros_like(r, dtype=bool)
        if len(terminals) > 0:
            terminals[-1] = True

        return MDPDataset(
            observations=obs.astype(np.float32),
            actions=action.astype(np.int64),
            rewards=r.astype(np.float32),
            terminals=terminals.astype(np.float32),
        )

    def fit(
            self,
            X,
            y,
            train_ratio: Optional[float] = None,
            random_state: int = 42,
            shuffle: bool = False,
            n_steps: Optional[int] = None,
            n_steps_per_epoch: Optional[int] = None,
            batch_size: Optional[int] = None,
            cql_config_kwargs: Optional[Dict[str, Any]] = None,
            device: Optional[str] = None,
    ):
        X = np.asarray(X, float)
        y = np.asarray(y, float)

        # 如果是多资产 (N, M)，先合成一个组合收益作为 reward
        if y.ndim == 2:
            # Equal-weight 组合，也可以换成你自己的权重 w
            reward_vec = y.mean(axis=1)
        else:
            reward_vec = y.reshape(-1)

        # 后面一律用 reward_vec 替代 y
        # ====== 划分训练集 ======
        tr_ratio = train_ratio if train_ratio is not None else self.train_ratio
        N = len(X)
        split = int(N * tr_ratio)
        X_train, reward_train = X[:split], reward_vec[:split]

        dataset = self._build_dataset(X_train, reward_train)

        # ====== 有效训练配置 ======
        eff_n_steps = n_steps if n_steps is not None else self.n_steps
        eff_n_steps_per_epoch = (
            n_steps_per_epoch if n_steps_per_epoch is not None else self.n_steps_per_epoch
        )
        eff_bs = batch_size if batch_size is not None else self.batch_size
        eff_bs = max(1, int(eff_bs))
        eff_device = device if device is not None else self.device

        # ====== 整理超参数（含 LLM 给的 dict） ======
        cfg_kwargs = dict(self.cql_config_kwargs)
        if cql_config_kwargs:
            cfg_kwargs.update(cql_config_kwargs)
        # 如果没有显式指定 batch_size，就用 eff_bs
        cfg_kwargs.setdefault("batch_size", eff_bs)

        # ====== 白名单过滤，防止 actor_learning_rate / weight_decay 之类乱入 ======
        # 为了绝对稳，这里只允许最安全的 gamma 和 batch_size
        valid_keys = {
            "batch_size",
            "gamma",
        }

        filtered_kwargs = {k: v for k, v in cfg_kwargs.items() if k in valid_keys}
        unused = {k: v for k, v in cfg_kwargs.items() if k not in valid_keys}
        if unused:
            print("[CQL WARNING] Ignored invalid CQLConfig keys:", unused)

        # ====== 创建 CQL 模型并训练 ======
        cql = d3rlpy.algos.DiscreteCQLConfig(
            compile_graph=True,
            **filtered_kwargs,
        ).create(device=eff_device)

        cql.fit(
            dataset,
            n_steps=eff_n_steps,
            n_steps_per_epoch=eff_n_steps_per_epoch,
        )

        wrapped = _CQLWrapper(cql)
        self.models_ = [("CQL_policy", wrapped)]
        return self.models_



