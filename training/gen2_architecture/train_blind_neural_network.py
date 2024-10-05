import numpy as np
from pathlib import Path
import sys, os

sys.path.append(str(Path(__file__).parent))
from training_data.load_data import (
    df_train_inputs_scaled,
    df_train_outputs_scaled,
    df_test_inputs_scaled,
    df_test_outputs_scaled,
    mean_inputs_scaled,
    cov_inputs_scaled,
)
import torch
from torch.utils.data import TensorDataset, DataLoader
import polars as pl
from typing import List
from tqdm import tqdm

N_inputs = len(df_train_inputs_scaled.columns)
N_outputs = len(df_train_outputs_scaled.columns)

cache_file = Path(__file__).parent / "nn-xxxlarge.pth"
n_hidden_layers = 5
width = 512
print("Cache file: ", cache_file)


# Define the model
class Net(torch.nn.Module):
    def __init__(self, mean_inputs_scaled, cov_inputs_scaled):
        super().__init__()

        self.mean_inputs_scaled = mean_inputs_scaled
        self.cov_inputs_scaled = cov_inputs_scaled
        self.inv_cov_inputs_scaled = torch.inverse(cov_inputs_scaled)
        self.N_inputs = len(mean_inputs_scaled)

        layers = [
            torch.nn.Linear(N_inputs, width),
            torch.nn.SiLU(),
        ]
        for i in range(n_hidden_layers):
            layers += [
                torch.nn.Linear(width, width),
                torch.nn.SiLU(),
            ]

        layers += [
            torch.nn.Linear(width, N_outputs),
        ]

        self.net = torch.nn.Sequential(*layers)

    def squared_mahalanobis_distance(self, x: torch.Tensor):
        return torch.sum(
            (x - self.mean_inputs_scaled) @ self.inv_cov_inputs_scaled * (x - self.mean_inputs_scaled),
            dim=1
        )

    def forward(self, x: torch.Tensor):
        ### First, evaluate the network normally
        y = self.net(x)
        y[:, 0] = y[:, 0] - self.squared_mahalanobis_distance(x=x) / (2 * N_inputs)
        ### Add in the squared Mahalanobis distance to the analysis_confidence logit, to ensure it
        # asymptotes to untrustworthy as the inputs get further from the training data

        ### Then, flip the inputs and evaluate the network again.
        # The goal here is to embed the invariant of "symmetry across alpha" into the network evaluation.

        x_flipped = x.clone()
        x_flipped[:, :8] = -1 * x[:, 8:16]  # switch kulfan_lower with a flipped kulfan_upper
        x_flipped[:, 8:16] = -1 * x[:, :8]  # switch kulfan_upper with a flipped kulfan_lower
        x_flipped[:, 16] = -1 * x[:, 16]  # flip kulfan_LE_weight
        x_flipped[:, 18] = -1 * x[:, 18]  # flip sin(2a)
        x_flipped[:, 23] = x[:, 24]  # flip xtr_upper with xtr_lower
        x_flipped[:, 24] = x[:, 23]  # flip xtr_lower with xtr_upper

        y_flipped = self.net(x_flipped)
        y_flipped[:, 0] = y_flipped[:, 0] - self.squared_mahalanobis_distance(x=x_flipped) / (2 * N_inputs)
        ### Add in the squared Mahalanobis distance to the analysis_confidence logit, to ensure it
        # asymptotes to untrustworthy as the inputs get further from the training data

        ### The resulting outputs will also be flipped, so we need to flip them back to their normal orientation
        y_unflipped = y_flipped.clone()
        y_unflipped[:, 1] = y_flipped[:, 1] * -1  # CL
        y_unflipped[:, 3] = y_flipped[:, 3] * -1  # CM
        y_unflipped[:, 4] = y_flipped[:, 5]  # switch Top_Xtr with Bot_Xtr
        y_unflipped[:, 5] = y_flipped[:, 4]  # switch Bot_Xtr with Top_Xtr

        # switch upper and lower Ret, H
        y_unflipped[:, 6 + 32 * 0: 6 + 32 * 2] = y_flipped[:, 6 + 32 * 3: 6 + 32 * 5]
        y_unflipped[:, 6 + 32 * 2: 6 + 32 * 3] = y_flipped[:, 6 + 32 * 5: 6 + 32 * 6] * -1  # ue/vinf
        y_unflipped[:, 6 + 32 * 3: 6 + 32 * 5] = y_flipped[:, 6 + 32 * 0: 6 + 32 * 2]
        y_unflipped[:, 6 + 32 * 5: 6 + 32 * 6] = y_flipped[:, 6 + 32 * 2: 6 + 32 * 3] * -1  # ue/vinf

        # switch upper_bl_ue/vinf with lower_bl_ue/vinf

        ### Then, average the two outputs to get the "symmetric" result
        y_fused = (y + y_unflipped) / 2
        # y_fused[:, 0] = torch.sigmoid(y_fused[:, 0])  # Analysis confidence, a binary variable
        y_fused[:, 4] = torch.clip(y_fused[:, 4].clone(), 0, 1)  # Top_Xtr clipped to range
        y_fused[:, 5] = torch.clip(y_fused[:, 5].clone(), 0, 1)  # Bot_Xtr clipped to range

        return y_fused


if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    net = Net(
        mean_inputs_scaled=torch.tensor(mean_inputs_scaled, dtype=torch.float32).to(device),
        cov_inputs_scaled=torch.tensor(cov_inputs_scaled, dtype=torch.float32).to(device),
    ).to(device)

    # Define the optimizer
    learning_rate = 1e-4
    optimizer = torch.optim.RAdam(net.parameters(), lr=learning_rate, weight_decay=3e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=50,
        verbose=True,
    )

    try:
        checkpoint = torch.load(cache_file)
        net.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        print("Model found, resuming training.")
    except FileNotFoundError:
        print("No existing model found, starting fresh.")

    # Define the data loader
    print(f"Preparing data...")

    batch_size = 256
    train_inputs = torch.tensor(
        df_train_inputs_scaled.to_numpy(),
        dtype=torch.float32,
    )
    train_outputs = torch.tensor(
        df_train_outputs_scaled.to_numpy(),
        dtype=torch.float32,
    )
    train_loader = DataLoader(
        dataset=TensorDataset(train_inputs, train_outputs),
        batch_size=batch_size,
        shuffle=True,
        # num_workers=4,
    )

    test_inputs = torch.tensor(
        df_test_inputs_scaled.to_numpy(),
        dtype=torch.float32,
    )
    test_outputs = torch.tensor(
        df_test_outputs_scaled.to_numpy(),
        dtype=torch.float32,
    )
    test_loader = DataLoader(
        dataset=TensorDataset(test_inputs, test_outputs),
        batch_size=8192,
        # num_workers=4,
    )

    # Prepare the loss function
    loss_weights = torch.ones(N_outputs, dtype=torch.float32).to(device)
    loss_weights[0] *= 0.005  # Analysis confidence
    loss_weights[1] *= 1  # CL
    loss_weights[2] *= 3  # ln(CD)
    loss_weights[3] *= 0.25  # CM
    loss_weights[4] *= 0.25  # Top Xtr
    loss_weights[5] *= 0.25  # Bot Xtr
    loss_weights[6:] *= 5e-3  # Lower the weight on all boundary layer outputs

    loss_weights = loss_weights / torch.sum(loss_weights) * 1000


    def loss_function(y_pred, y_data, return_individual_loss_components=False):
        # For data with NaN, overwrite the data with the prediction. This essentially makes the model ignore NaN data,
        # since the gradient of the loss with respect to parameters is zero when the data is NaN.
        y_data = torch.where(
            torch.isnan(y_data),
            y_pred,
            y_data
        )

        analysis_confidence_loss = torch.mean(
            torch.nn.functional.binary_cross_entropy_with_logits(
                input=y_pred[:, 0:1],
                target=y_data[:, 0:1],
                reduction='none',
            ),
            dim=0
        )
        # other_loss_components = torch.mean(
        #     (y_pred[:, 1:] - y_data[:, 1:]) ** 2,
        #     dim=0
        # )

        other_loss_components = torch.mean(
            torch.nn.functional.huber_loss(
                y_pred[:, 1:], y_data[:, 1:],
                reduction='none',
                delta=0.05
            ),
            dim=0
        )

        # other_loss_components = torch.mean(
        #     torch.nn.functional.mse_loss(
        #         y_pred[:, 1:], y_data[:, 1:],
        #         reduction='none',
        #     ),
        #     dim=0
        # )

        unweighted_loss_components = torch.concatenate([
            analysis_confidence_loss,
            other_loss_components
        ], dim=0)

        weighted_loss_components = unweighted_loss_components * loss_weights

        if return_individual_loss_components:
            return weighted_loss_components
        else:
            return torch.sum(weighted_loss_components)


    # raise Exception
    print(f"Training...")

    n_batches_per_epoch = len(train_loader)

    num_epochs = 10 ** 9  # Effectively loop until manually stopped
    for epoch in range(num_epochs):
        # Put the model in training mode
        net.train()

        loss_from_each_training_batch = []

        # for x, y_data in tqdm(train_loader):
        for x, y_data in train_loader:

            x = x.to(device)
            y_data = y_data.to(device)

            loss = loss_function(
                y_pred=net(x),
                y_data=y_data
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_from_each_training_batch.append(loss.detach())

        train_loss = torch.mean(torch.stack(loss_from_each_training_batch, dim=0), dim=0)

        # Put the model in evaluation mode
        net.eval()

        loss_components_from_each_test_batch = []
        mae_from_each_test_batch = []

        for i, (x, y_data) in enumerate(test_loader):
            with torch.no_grad():
                x = x.to(device)
                y_data = y_data.to(device)

                y_pred = net(x)

                loss_components = loss_function(
                    y_pred=y_pred,
                    y_data=y_data,
                    return_individual_loss_components=True
                )

                loss_components_from_each_test_batch.append(loss_components)

                y_pred[:, 0] = torch.sigmoid(y_pred[:, 0])  # Analysis confidence, a binary variable

                mae_from_each_test_batch.append(
                    torch.nanmean(torch.abs(y_pred - y_data), dim=0)
                )

        test_loss_components = torch.mean(torch.stack(loss_components_from_each_test_batch, dim=0), dim=0)
        test_loss = torch.sum(test_loss_components)
        test_residual_mae = torch.nanmean(torch.stack(mae_from_each_test_batch, dim=0), dim=0)

        labeled_maes = {
            "analysis_confidence": test_residual_mae[0],
            "CL"                 : test_residual_mae[1] / 2,
            "ln_CD"              : test_residual_mae[2] * 2,
            "CM"                 : test_residual_mae[3] / 20,
            "Top_Xtr"            : test_residual_mae[4],
            "Bot_Xtr"            : test_residual_mae[5],
        }
        print(
            f"Epoch: {epoch} | Train Loss: {train_loss.item():.6g} | Test Loss: {test_loss.item():.6g} | "
            + " | ".join(
                [
                    f"{k}: {v:.6g}"
                    for k, v in labeled_maes.items()
                ]
            )
        )
        loss_argsort = torch.argsort(test_loss_components, descending=True)
        print(f"Loss contributors: ")
        for i in loss_argsort[:10]:
            print(f"\t{df_train_outputs_scaled.columns[i]:25}: {test_loss_components[i].item():.6g}")

        scheduler.step(test_loss)

        torch.save({
            'model_state_dict'    : net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, cache_file)
