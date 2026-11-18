"""Small CPU Torch classifiers; probabilities always refer to CKD=1."""

import torch
from torch import nn
from torch.nn import functional as F


def checked(X, y=None):
    X = torch.as_tensor(X, dtype=torch.float64)
    if X.ndim != 2 or not X.numel() or not torch.isfinite(X).all():
        raise ValueError("Expected a finite nonempty feature matrix")
    if y is not None:
        y = torch.as_tensor(y, dtype=torch.float64)
        if y.shape != (len(X),) or not ((y == 0) | (y == 1)).all():
            raise ValueError("Expected binary labels matching rows")
    return X, y


class Dummy:
    def __init__(self, strategy="most_frequent"):
        if strategy not in ("most_frequent", "always_ckd"):
            raise ValueError(strategy)
        self.strategy = strategy

    def fit(self, X, y):
        X, y = checked(X, y)
        self.width = X.shape[1]
        self.value = float(self.strategy == "always_ckd" or y.mean() > 0.5)
        return self

    def predict_proba(self, X):
        X, _ = checked(X)
        if not hasattr(self, "width") or X.shape[1] != self.width:
            raise ValueError("Unfitted or incompatible input")
        return torch.full((len(X),), self.value, dtype=X.dtype)


class LogisticRegression(Dummy):
    def __init__(self, C=1.0, class_weight=None, max_iter=1000):
        if C <= 0 or class_weight not in (None, "balanced"):
            raise ValueError("Invalid parameters")
        self.C = C
        self.class_weight = class_weight
        self.max_iter = max_iter

    def fit(self, X, y):
        X, y = checked(X, y)
        self.width = X.shape[1]
        self.net = nn.Linear(self.width, 1).double()
        nn.init.zeros_(self.net.weight)
        nn.init.zeros_(self.net.bias)
        sample_weights = torch.ones_like(y)
        if self.class_weight:
            class_counts = torch.bincount(y.long(), minlength=2)
            if (class_counts == 0).any():
                raise ValueError("Balanced fitting requires both classes")
            sample_weights = len(y) / (2 * class_counts[y.long()])

        optimizer = torch.optim.LBFGS(
            self.net.parameters(),
            max_iter=self.max_iter,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad()
            losses = F.binary_cross_entropy_with_logits(
                self.net(X).flatten(), y, reduction="none"
            )
            objective = (losses * sample_weights).sum() / sample_weights.sum()
            objective += self.net.weight.square().sum() / (
                2 * self.C * sample_weights.sum()
            )
            objective.backward()
            return objective

        optimizer.step(closure)
        self.objective = float(closure().detach())
        self.gradient_norm = max(
            float(parameter.grad.abs().max()) for parameter in self.net.parameters()
        )
        self.converged = self.gradient_norm < 1e-4
        return self

    def predict_proba(self, X):
        X, _ = checked(X)
        if not hasattr(self, "net") or X.shape[1] != self.width:
            raise ValueError("Unfitted or incompatible input")
        with torch.no_grad():
            return self.net(X).flatten().sigmoid()


class RandomForest(Dummy):
    def __init__(
        self,
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight=None,
        seed=42,
    ):
        if (
            n_estimators < 1
            or min_samples_leaf < 1
            or (max_depth is not None and max_depth < 1)
        ):
            raise ValueError("Invalid tree size")
        if max_features != "sqrt" and not (
            isinstance(max_features, (float, int)) and 0 < max_features <= 1
        ):
            raise ValueError("Invalid max_features")
        if class_weight not in (None, "balanced"):
            raise ValueError("Invalid weighting")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.class_weight = class_weight
        self.seed = seed

    def fit(self, X, y):
        X, y = checked(X, y)
        self.width = X.shape[1]
        self.trees = []
        self.importances = torch.zeros(self.width)
        generator = torch.Generator().manual_seed(self.seed)
        sample_weights = torch.ones_like(y)
        if self.class_weight:
            class_counts = torch.bincount(y.long(), minlength=2)
            if (class_counts == 0).any():
                raise ValueError("Both classes required")
            sample_weights = len(y) / (2 * class_counts[y.long()])
        selected_feature_count = max(
            1,
            int(
                self.width**0.5
                if self.max_features == "sqrt"
                else self.width * self.max_features
            ),
        )

        def build(row_ids, depth):
            node_weights = sample_weights[row_ids]
            labels = y[row_ids]
            positive_rate = (node_weights * labels).sum() / node_weights.sum()
            best_split = None
            best_gain = 0.0
            if (
                positive_rate == 0
                or positive_rate == 1
                or (self.max_depth is not None and depth >= self.max_depth)
            ):
                return float(positive_rate)

            feature_ids = torch.randperm(self.width, generator=generator)[
                :selected_feature_count
            ].tolist()
            for feature_index in sorted(feature_ids):
                order = torch.argsort(X[row_ids, feature_index], stable=True)
                values = X[row_ids[order], feature_index]
                cumulative_weight = node_weights[order].cumsum(0)
                cumulative_positive_weight = (
                    node_weights[order] * labels[order]
                ).cumsum(0)
                cut_positions = torch.arange(1, len(row_ids))
                valid = (
                    (values[:-1] < values[1:])
                    & (cut_positions >= self.min_samples_leaf)
                    & (len(row_ids) - cut_positions >= self.min_samples_leaf)
                )
                if not valid.any():
                    continue

                left_weight = cumulative_weight[:-1]
                right_weight = node_weights.sum() - cumulative_weight[:-1]
                left_rate = cumulative_positive_weight[:-1] / left_weight
                right_rate = (
                    cumulative_positive_weight[-1] - cumulative_positive_weight[:-1]
                ) / right_weight
                gains = 2 * positive_rate * (1 - positive_rate) - (
                    left_weight * 2 * left_rate * (1 - left_rate)
                    + right_weight * 2 * right_rate * (1 - right_rate)
                ) / node_weights.sum()
                gains[~valid] = -float("inf")
                split_index = int(gains.argmax())
                gain = float(gains[split_index])
                threshold = float(
                    values[split_index] / 2 + values[split_index + 1] / 2
                )
                left = X[row_ids, feature_index] <= threshold
                if (
                    min(int(left.sum()), int((~left).sum()))
                    < self.min_samples_leaf
                ):
                    continue
                if gain > best_gain + 1e-14:
                    best_split = (feature_index, threshold, left)
                    best_gain = gain

            if best_split is None:
                return float(positive_rate)
            feature_index, threshold, left = best_split
            self.importances[feature_index] += best_gain * float(node_weights.sum())
            return (
                feature_index,
                threshold,
                build(row_ids[left], depth + 1),
                build(row_ids[~left], depth + 1),
            )

        for _ in range(self.n_estimators):
            bootstrap_ids = torch.randint(
                len(X), (len(X),), generator=generator
            )
            self.trees.append(build(bootstrap_ids, 0))
        if self.importances.sum() > 0:
            self.importances /= self.importances.sum()
        return self

    def predict_proba(self, X):
        X, _ = checked(X)
        if not hasattr(self, "trees") or X.shape[1] != self.width:
            raise ValueError("Unfitted or incompatible input")

        def predict(tree, row):
            while isinstance(tree, tuple):
                tree = tree[2] if row[tree[0]] <= tree[1] else tree[3]
            return tree

        tree_predictions = [
            [predict(tree, row) for tree in self.trees] for row in X
        ]
        return torch.tensor(tree_predictions, dtype=X.dtype).mean(1)


def binary_loss(logits, y, loss="bce", pos_weight=1.0):
    if loss == "mse":
        return F.mse_loss(logits.sigmoid(), y)
    if loss == "weighted_bce":
        return F.binary_cross_entropy_with_logits(
            logits, y, pos_weight=logits.new_tensor(pos_weight)
        )
    binary_cross_entropy = F.binary_cross_entropy_with_logits(
        logits, y, reduction="none"
    )
    if loss == "focal":
        return (
            (1 - torch.exp(-binary_cross_entropy)).square() * binary_cross_entropy
        ).mean()
    if loss != "bce":
        raise ValueError(loss)
    return binary_cross_entropy.mean()


class NeuralNetwork(LogisticRegression):
    """Fixed-epoch estimator; the runner selects epochs using training-only data."""

    def __init__(
        self,
        hidden=(16,),
        epochs=200,
        batch_size=16,
        weight_decay=0.0,
        loss="bce",
        seed=42,
    ):
        self.hidden = hidden
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.loss = loss
        self.seed = seed

    def fit(self, X, y, validation=None, patience=20):
        X, y = checked(X, y)
        self.width = X.shape[1]
        if (
            self.epochs < 1
            or self.batch_size < 1
            or self.weight_decay < 0
            or not self.hidden
            or any(width < 1 for width in self.hidden)
        ):
            raise ValueError("Invalid NN configuration")
        if self.loss not in ("bce", "weighted_bce", "mse", "focal"):
            raise ValueError("Invalid loss")
        if validation is not None:
            validation_features, validation_labels = checked(*validation)
            if validation_features.shape[1] != self.width:
                raise ValueError("Validation width mismatch")
        if not 0 < y.sum() < len(y):
            raise ValueError("Both classes required")

        positive_weight = float((len(y) - y.sum()) / y.sum())
        with torch.random.fork_rng():
            torch.manual_seed(self.seed)
            layers = []
            input_width = self.width
            for hidden_width in self.hidden:
                layers.extend([nn.Linear(input_width, hidden_width), nn.ReLU()])
                input_width = hidden_width
            layers.append(nn.Linear(input_width, 1))
            self.net = nn.Sequential(*layers).double()
            optimizer = torch.optim.Adam(
                self.net.parameters(), lr=1e-3, weight_decay=self.weight_decay
            )
            self.history = []
            self.validation_history = []
            self.best_epoch = 1
            best_validation_loss = float("inf")
            stale_epochs = 0
            for epoch in range(self.epochs):
                order = torch.randperm(len(X))
                for batch_ids in order.split(self.batch_size):
                    optimizer.zero_grad()
                    loss_value = binary_loss(
                        self.net(X[batch_ids]).flatten(),
                        y[batch_ids],
                        self.loss,
                        positive_weight,
                    )
                    loss_value.backward()
                    optimizer.step()
                with torch.no_grad():
                    training_loss = binary_loss(
                        self.net(X).flatten(), y, self.loss, positive_weight
                    )
                self.history.append(float(training_loss))

                if validation is not None:
                    with torch.no_grad():
                        validation_loss = float(
                            binary_loss(
                                self.net(validation_features).flatten(),
                                validation_labels,
                                self.loss,
                                positive_weight,
                            )
                        )
                    self.validation_history.append(validation_loss)
                    if validation_loss < best_validation_loss - 1e-8:
                        best_validation_loss = validation_loss
                        self.best_epoch = epoch + 1
                        stale_epochs = 0
                    else:
                        stale_epochs += 1
                    if stale_epochs >= patience:
                        break
                else:
                    self.best_epoch = epoch + 1
        return self
