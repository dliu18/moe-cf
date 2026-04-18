from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pickle

ArrayLike = Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]]


class ExpPerceptronRegressor:
    """
    y_hat = c + b * exp(- a^T p)

    Parameters:
      - a: vector (d,)
      - b: scalar
      - c: scalar
    """

    def __init__(
        self,
        dim: int,
        *,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
        seed: Optional[int] = 0,
    ):
        self.dim = int(dim)
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.dtype = dtype

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Parameters (unconstrained)
        self.a = nn.Parameter(torch.zeros(self.dim, device=self.device, dtype=self.dtype))
        self.b = nn.Parameter(torch.tensor(0.0, device=self.device, dtype=self.dtype))
        self.c = nn.Parameter(torch.tensor(0.0, device=self.device, dtype=self.dtype))

        self._params = [self.a, self.b, self.c]
        self.loss_history: List[float] = []

    # ----------------------------
    # Core model
    # ----------------------------
    def _forward(self, P: torch.Tensor) -> torch.Tensor:
        """
        P: (n, d)
        returns y_hat: (n,)
        """
        if P.ndim != 2 or P.shape[1] != self.dim:
            raise ValueError(f"P must have shape (n, {self.dim}), got {tuple(P.shape)}")
        # (n,) = (n,d) @ (d,)
        dot = P @ self.a
        y_hat = self.c + self.b * torch.exp(-dot)
        return y_hat

    # ----------------------------
    # Utilities: data conversion
    # ----------------------------
    def _to_tensor_P(self, P: ArrayLike) -> torch.Tensor:
        P_np = np.asarray(P, dtype=np.float32)
        if P_np.ndim == 1:
            P_np = P_np.reshape(1, -1)
        if P_np.shape[1] != self.dim:
            raise ValueError(f"Expected P with dim={self.dim}, got {P_np.shape[1]}")
        return torch.tensor(P_np, device=self.device, dtype=self.dtype)

    def _to_tensor_y(self, y: ArrayLike) -> torch.Tensor:
        y_np = np.asarray(y, dtype=np.float32).reshape(-1)
        return torch.tensor(y_np, device=self.device, dtype=self.dtype)

    # ----------------------------
    # Training
    # ----------------------------
    def fit(
        self,
        P_train: ArrayLike,
        y_train: ArrayLike,
        *,
        lr: float = 1e-2,
        weight_decay: float = 0.0,
        epochs: int = 2000,
        batch_size: Optional[int] = None,
        loss: str = "mse",
        verbose_every: int = 200,
        clip_grad_norm: Optional[float] = None,
    ) -> "ExpPerceptronRegressor":
        """
        Trains parameters a, b, c by gradient descent.

        loss: "mse" or "huber"
        batch_size: None => full batch
        """
        P = self._to_tensor_P(P_train)
        y = self._to_tensor_y(y_train)

        if P.shape[0] != y.shape[0]:
            raise ValueError(f"Mismatch: P has {P.shape[0]} rows but y has {y.shape[0]} items")

        if loss.lower() == "mse":
            criterion = nn.MSELoss()
        elif loss.lower() == "huber":
            criterion = nn.SmoothL1Loss()
        else:
            raise ValueError("loss must be 'mse' or 'huber'")

        opt = torch.optim.Adam(self._params, lr=lr, weight_decay=weight_decay)

        n = P.shape[0]
        bs = n if (batch_size is None or batch_size >= n) else int(batch_size)

        self.loss_history = []

        for epoch in range(1, epochs + 1):
            # mini-batch shuffle
            if bs < n:
                idx = torch.randperm(n, device=self.device)
                P_epoch = P[idx]
                y_epoch = y[idx]
            else:
                P_epoch = P
                y_epoch = y

            epoch_losses = []

            for start in range(0, n, bs):
                end = min(start + bs, n)
                Pb = P_epoch[start:end]
                yb = y_epoch[start:end]

                opt.zero_grad(set_to_none=True)
                y_hat = self._forward(Pb)
                L = criterion(y_hat, yb)
                L.backward()

                if clip_grad_norm is not None:
                    nn.utils.clip_grad_norm_(self._params, max_norm=clip_grad_norm)

                opt.step()
                epoch_losses.append(float(L.detach().cpu().item()))

            mean_loss = float(np.mean(epoch_losses))
            self.loss_history.append(mean_loss)

            if verbose_every > 0 and (epoch == 1 or epoch % verbose_every == 0 or epoch == epochs):
                print(
                    f"epoch {epoch:5d}/{epochs} | loss={mean_loss:.6g} | "
                    f"b={self.b.item():.4g} c={self.c.item():.4g} ||a||={self.a.norm().item():.4g}"
                )

        return self

    # ----------------------------
    # Prediction & metrics
    # ----------------------------
    @torch.no_grad()
    def predict(self, P: ArrayLike) -> np.ndarray:
        """
        Returns predictions as a numpy array of shape (n,).
        """
        self._check_trained()
        Pt = self._to_tensor_P(P)
        y_hat = self._forward(Pt)
        return y_hat.detach().cpu().numpy()

    def r2(self, P: ArrayLike, y_true: ArrayLike) -> float:
        """
        R^2 = 1 - SSE/SST
        """
        self._check_trained()
        y_true_np = np.asarray(y_true, dtype=np.float64).reshape(-1)
        y_pred_np = self.predict(P).astype(np.float64)

        if y_true_np.shape[0] != y_pred_np.shape[0]:
            raise ValueError("y_true and predictions have different lengths")

        sse = float(np.sum((y_true_np - y_pred_np) ** 2))
        sst = float(np.sum((y_true_np - np.mean(y_true_np)) ** 2))
        if sst == 0.0:
            # all y are identical; R^2 is not well-defined
            return float("nan")
        return 1.0 - (sse / sst)

    # ----------------------------
    # Plotting
    # ----------------------------
    def plot_loss(self, *, logy: bool = False, save: bool = True, filename: str = "") -> None:
        if not self.loss_history:
            raise RuntimeError("No loss history found. Call fit() first.")
        plt.figure()
        plt.plot(self.loss_history)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training loss")
        if logy:
            plt.yscale("log")
        plt.tight_layout()
        if save:
            plt.savefig(f"plots/regression/{filename}.pdf", bbox_inches="tight")

    # ----------------------------
    # Introspection
    # ----------------------------
    def parameters_numpy(self) -> Tuple[np.ndarray, float, float]:
        """
        Returns (a, b, c) as numpy/scalars.
        """
        self._check_trained()
        return (
            self.a.detach().cpu().numpy().copy(),
            float(self.b.detach().cpu().item()),
            float(self.c.detach().cpu().item()),
        )

    def _check_trained(self) -> None:
        # In this simple setup params always exist, but we keep the guard anyway.
        if any(p is None for p in [self.a, self.b, self.c]):
            raise RuntimeError("Model parameters not initialized.")


class PolynomialRegressor:
    """
    Polynomial regression of y on p:

        y_hat = w_0 + sum_{k=1}^degree sum_{j=1}^dim w_{k,j} * p_j^k

    Parameters:
      - degree: maximum polynomial degree
    """

    def __init__(
        self,
        dim: int,
        degree: int,
        *,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
        seed: Optional[int] = 0,
    ):
        if degree < 1:
            raise ValueError("degree must be >= 1")

        self.dim = int(dim)
        self.degree = int(degree)
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.dtype = dtype

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Number of features: 1 (bias) + dim * degree
        self.num_features = 1 + self.dim * self.degree

        # Linear weights over polynomial features
        self.W = nn.Parameter(
            torch.zeros(self.num_features, device=self.device, dtype=self.dtype)
        )

        self._params = [self.W]
        self.loss_history: List[float] = []

    # ----------------------------
    # Feature construction
    # ----------------------------
    def _poly_features(self, P: torch.Tensor) -> torch.Tensor:
        """
        P: (n, d)
        returns Phi: (n, 1 + d * degree)
        """
        if P.ndim != 2 or P.shape[1] != self.dim:
            raise ValueError(f"P must have shape (n, {self.dim}), got {tuple(P.shape)}")

        feats = [torch.ones((P.shape[0], 1), device=P.device, dtype=P.dtype)]
        for k in range(1, self.degree + 1):
            feats.append(P ** k)

        return torch.cat(feats, dim=1)

    # ----------------------------
    # Core model
    # ----------------------------
    def _forward(self, P: torch.Tensor) -> torch.Tensor:
        Phi = self._poly_features(P)
        return Phi @ self.W

    # ----------------------------
    # Utilities: data conversion
    # ----------------------------
    def _to_tensor_P(self, P: ArrayLike) -> torch.Tensor:
        P_np = np.asarray(P, dtype=np.float32)
        if P_np.ndim == 1:
            P_np = P_np.reshape(1, -1)
        if P_np.shape[1] != self.dim:
            raise ValueError(f"Expected P with dim={self.dim}, got {P_np.shape[1]}")
        return torch.tensor(P_np, device=self.device, dtype=self.dtype)

    def _to_tensor_y(self, y: ArrayLike) -> torch.Tensor:
        y_np = np.asarray(y, dtype=np.float32).reshape(-1)
        return torch.tensor(y_np, device=self.device, dtype=self.dtype)

    # ----------------------------
    # Training
    # ----------------------------
    def fit(
        self,
        P_train: ArrayLike,
        y_train: ArrayLike,
        *,
        lr: float = 1e-2,
        weight_decay: float = 0.0,
        epochs: int = 2000,
        batch_size: Optional[int] = None,
        loss: str = "mse",
        verbose_every: int = 200,
        clip_grad_norm: Optional[float] = None,
    ) -> "PolynomialRegressor":

        P = self._to_tensor_P(P_train)
        y = self._to_tensor_y(y_train)

        if P.shape[0] != y.shape[0]:
            raise ValueError(
                f"Mismatch: P has {P.shape[0]} rows but y has {y.shape[0]} items"
            )

        if loss.lower() == "mse":
            criterion = nn.MSELoss()
        elif loss.lower() == "huber":
            criterion = nn.SmoothL1Loss()
        else:
            raise ValueError("loss must be 'mse' or 'huber'")

        opt = torch.optim.Adam(self._params, lr=lr, weight_decay=weight_decay)

        n = P.shape[0]
        bs = n if (batch_size is None or batch_size >= n) else int(batch_size)

        self.loss_history = []

        for epoch in range(1, epochs + 1):
            if bs < n:
                idx = torch.randperm(n, device=self.device)
                P_epoch = P[idx]
                y_epoch = y[idx]
            else:
                P_epoch = P
                y_epoch = y

            epoch_losses = []

            for start in range(0, n, bs):
                end = min(start + bs, n)
                Pb = P_epoch[start:end]
                yb = y_epoch[start:end]

                opt.zero_grad(set_to_none=True)
                y_hat = self._forward(Pb)
                L = criterion(y_hat, yb)
                L.backward()

                if clip_grad_norm is not None:
                    nn.utils.clip_grad_norm_(self._params, max_norm=clip_grad_norm)

                opt.step()
                epoch_losses.append(float(L.detach().cpu().item()))

            mean_loss = float(np.mean(epoch_losses))
            self.loss_history.append(mean_loss)

            if verbose_every > 0 and (
                epoch == 1 or epoch % verbose_every == 0 or epoch == epochs
            ):
                print(
                    f"epoch {epoch:5d}/{epochs} | loss={mean_loss:.6g} | ||W||={self.W.norm().item():.4g}"
                )

        return self

    # ----------------------------
    # Prediction & metrics
    # ----------------------------
    @torch.no_grad()
    def predict(self, P: ArrayLike) -> np.ndarray:
        self._check_trained()
        Pt = self._to_tensor_P(P)
        y_hat = self._forward(Pt)
        return y_hat.detach().cpu().numpy()

    # ----------------------------
    # Optimization on simplex
    # ----------------------------
    def find_extrema_on_simplex(
        self,
        *,
        steps: int = 2000,
        lr: float = 1e-2,
        n_restarts: int = 10,
        maximize: bool = True,
        verbose: bool = False,
    ):
        """
        Optimize the polynomial regression over the simplex:

            p >= 0
            sum(p) = 1

        Returns:
            p_opt, y_opt
        """

        self._check_trained()

        def project_simplex(v):
            """Euclidean projection onto simplex."""
            v = np.asarray(v)
            u = np.sort(v)[::-1]
            cssv = np.cumsum(u)
            rho = np.nonzero(u * np.arange(1, len(v)+1) > (cssv - 1))[0][-1]
            theta = (cssv[rho] - 1) / (rho + 1)
            return np.maximum(v - theta, 0)

        best_p = None
        best_val = None

        for restart in range(n_restarts):

            # random start on simplex
            p = np.random.rand(self.dim)
            p = p / p.sum()
            p_t = torch.tensor(p, dtype=self.dtype, device=self.device, requires_grad=True)

            opt = torch.optim.SGD([p_t], lr=lr)

            for step in range(steps):

                opt.zero_grad()

                y_hat = self._forward(p_t.unsqueeze(0)).squeeze()

                loss = -y_hat if maximize else y_hat
                loss.backward()
                opt.step()

                # projection
                with torch.no_grad():
                    p_np = p_t.detach().cpu().numpy()
                    p_np = project_simplex(p_np)
                    p_t.copy_(torch.tensor(p_np, device=self.device, dtype=self.dtype))

            with torch.no_grad():
                final_val = float(self._forward(p_t.unsqueeze(0)).item())
                final_p = p_t.detach().cpu().numpy().copy()

            if best_val is None:
                best_val = final_val
                best_p = final_p
            else:
                if maximize and final_val > best_val:
                    best_val = final_val
                    best_p = final_p
                if not maximize and final_val < best_val:
                    best_val = final_val
                    best_p = final_p

            if verbose:
                print(f"Restart {restart}: value={final_val:.6g}")

        return best_p, best_val

    def r2(self, P: ArrayLike, y_true: ArrayLike) -> float:
        self._check_trained()
        y_true_np = np.asarray(y_true, dtype=np.float64).reshape(-1)
        y_pred_np = self.predict(P).astype(np.float64)

        if y_true_np.shape[0] != y_pred_np.shape[0]:
            raise ValueError("y_true and predictions have different lengths")

        sse = float(np.sum((y_true_np - y_pred_np) ** 2))
        sst = float(np.sum((y_true_np - np.mean(y_true_np)) ** 2))
        if sst == 0.0:
            return float("nan")
        return 1.0 - (sse / sst)

    # ----------------------------
    # Optimization on simplex
    # ----------------------------
    def find_extrema_on_simplex(
        self,
        *,
        steps: int = 2000,
        lr: float = 1e-2,
        n_restarts: int = 10,
        maximize: bool = True,
        verbose: bool = False,
    ):
        """
        Optimize the polynomial regression over the simplex:

            p >= 0
            sum(p) = 1

        Returns:
            p_opt, y_opt
        """

        self._check_trained()

        def project_simplex(v):
            """Euclidean projection onto simplex."""
            v = np.asarray(v)
            u = np.sort(v)[::-1]
            cssv = np.cumsum(u)
            rho = np.nonzero(u * np.arange(1, len(v)+1) > (cssv - 1))[0][-1]
            theta = (cssv[rho] - 1) / (rho + 1)
            return np.maximum(v - theta, 0)

        best_p = None
        best_val = None

        for restart in range(n_restarts):

            # random start on simplex
            p = np.random.rand(self.dim)
            p = p / p.sum()
            p_t = torch.tensor(p, dtype=self.dtype, device=self.device, requires_grad=True)

            opt = torch.optim.SGD([p_t], lr=lr)

            for step in range(steps):

                opt.zero_grad()

                y_hat = self._forward(p_t.unsqueeze(0)).squeeze()

                loss = -y_hat if maximize else y_hat
                loss.backward()
                opt.step()

                # projection
                with torch.no_grad():
                    p_np = p_t.detach().cpu().numpy()
                    p_np = project_simplex(p_np)
                    p_t.copy_(torch.tensor(p_np, device=self.device, dtype=self.dtype))

            with torch.no_grad():
                final_val = float(self._forward(p_t.unsqueeze(0)).item())
                final_p = p_t.detach().cpu().numpy().copy()

            if best_val is None:
                best_val = final_val
                best_p = final_p
            else:
                if maximize and final_val > best_val:
                    best_val = final_val
                    best_p = final_p
                if not maximize and final_val < best_val:
                    best_val = final_val
                    best_p = final_p

            if verbose:
                print(f"Restart {restart}: value={final_val:.6g}")

        return best_p, best_val

    # ----------------------------
    # Plotting
    # ----------------------------
    def plot_loss(self, *, logy: bool = False, save: bool = True, filename: str = "") -> None:
        if not self.loss_history:
            raise RuntimeError("No loss history found. Call fit() first.")
        plt.figure()
        plt.plot(self.loss_history)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training loss")
        if logy:
            plt.yscale("log")
        plt.tight_layout()
        if save:
            plt.savefig(f"plots/regression/{filename}.pdf", bbox_inches="tight")

    # ----------------------------
    # Introspection
    # ----------------------------
    def parameters_numpy(self) -> np.ndarray:
        """
        Returns polynomial weights W as a numpy array.
        """
        self._check_trained()
        return self.W.detach().cpu().numpy().copy()

    def _check_trained(self) -> None:
        if self.W is None:
            raise RuntimeError("Model parameters not initialized.")



# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    exp_name = "expected_coef_all_groups_full_aug"
    filename = f"results/mixing_results_by_group_{exp_name}.pickle"
    metric_name = "precision"

    with open(filename, "rb") as pickleFile:
        mixing_results_by_group = pickle.load(pickleFile)

    for base_label in mixing_results_by_group:
        assert len(mixing_results_by_group[base_label]["validation ps"]) == len(mixing_results_by_group[base_label]["validation metrics"])
        print(f"{base_label}: {len(mixing_results_by_group[base_label]['validation ps'])} trials")
        
        ps = mixing_results_by_group[base_label]["validation ps"]
        group_labels = set(ps[0].keys())
        P = np.array([
            [p[aug_label] for aug_label in group_labels]
            for p in ps
            ])
        assert np.all([group_labels == set(p.keys()) for p in ps])
        
        metric_dicts = mixing_results_by_group[base_label]["validation metrics"]
        metric_names = set(metric_dicts[0].keys())
        assert np.all([metric_names == set(metric_dict.keys()) for metric_dict in metric_dicts])
        y = np.array([metric_dict[metric_name] for metric_dict in metric_dicts])
        
        model = ExpPerceptronRegressor(dim=len(group_labels), seed=0)
        model.fit(P, y, epochs=1000, lr=1e-3, verbose_every=1000, loss="mse")
        print("Train R2:", model.r2(P, y))

        model.plot_loss(logy=True, save=True, filename=base_label)

