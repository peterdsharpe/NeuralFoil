import numpy as np
from pathlib import Path
import sys, os

sys.path.append(str(Path(__file__).parent))
from training_data.load_data import (
    df_train_inputs_scaled,
    df_train_outputs_scaled,
    df_test_inputs_scaled,
    df_test_outputs_scaled,
)
import torch
from torch.utils.data import TensorDataset, DataLoader
import polars as pl
from typing import List
from tqdm import tqdm

N_inputs = len(df_train_inputs_scaled.columns)
N_outputs = len(df_train_outputs_scaled.columns)

cache_file = Path(__file__).parent / "nn-xxxlarge.pth"
print("Cache file: ", cache_file)

# Define the model
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()

        width = 512

        self.net = torch.nn.Sequential(
            torch.nn.Linear(N_inputs, width),
            torch.nn.Tanh(),

            torch.nn.Linear(width, width),
            torch.nn.Tanh(),
            torch.nn.Linear(width, width),
            torch.nn.Tanh(),
            torch.nn.Linear(width, width),
            torch.nn.Tanh(),
            torch.nn.Linear(width, width),
            torch.nn.Tanh(),

            torch.nn.Linear(width, N_outputs),
        )

    def forward(self, x: torch.Tensor):
        ### First, evaluate the network normally
        y = self.net(x)

        ### Then, flip the inputs and evaluate the network again.
        # The goal here is to embed the invariant of "symmetry across alpha" into the network evaluation.

        x_flipped = x.clone()
        x_flipped[:, :8] = x[:, 8:16] * -1  # switch kulfan_lower with a flipped kulfan_upper
        x_flipped[:, 8:16] = x[:, :8] * -1  # switch kulfan_upper with a flipped kulfan_lower
        x_flipped[:, 16] *= -1  # flip kulfan_LE_weight
        x_flipped[:, 18] *= -1  # flip sin(2a)
        x_flipped[:, 23] = x[:, 24]  # flip xtr_upper with xtr_lower
        x_flipped[:, 24] = x[:, 23]  # flip xtr_lower with xtr_upper

        y_flipped = self.net(x_flipped)

        ### The resulting outputs will also be flipped, so we need to flip them back to their normal orientation
        y_unflipped = y_flipped.clone()
        y_unflipped[:, 1] *= -1  # CL
        y_unflipped[:, 3] *= -1  # CM
        y_unflipped[:, 4] = y_flipped[:, 5]  # switch Top_Xtr with Bot_Xtr
        y_unflipped[:, 5] = y_flipped[:, 4]  # switch Bot_Xtr with Top_Xtr

        # switch upper and lower Ret, H
        y_unflipped[:, 6:6 + 32 * 2] = y_flipped[:, 6 + 32 * 3: 6 + 32 * 5]
        y_unflipped[:, 6 + 32 * 3: 6 + 32 * 5] = y_flipped[:, 6:6 + 32 * 2]

        # switch upper_bl_ue/vinf with lower_bl_ue/vinf
        y_unflipped[:, 6 + 32 * 2: 6 + 32 * 3] = -1 * y_flipped[:, 6 + 32 * 5: 6 + 32 * 6]
        y_unflipped[:, 6 + 32 * 5: 6 + 32 * 6] = -1 * y_flipped[:, 6 + 32 * 2: 6 + 32 * 3]

        ### Then, average the two outputs to get the "symmetric" result
        y_fused = (y + y_unflipped) / 2
        y_fused[:, 0] = torch.sigmoid(y_fused[:, 0])  # Analysis confidence, a binary variable

        return y_fused


if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    net = Net().to(device)

    try:
        net.load_state_dict(torch.load(cache_file))
        print("Model found, resuming training.")
    except FileNotFoundError:
        print("No existing model found, starting fresh.")

    # Define the optimizer
    learning_rate = 1e-3
    optimizer = torch.optim.RAdam(net.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=10,
        verbose=True,
        min_lr=0,
    )

    # Define the data loader
    print(f"Preparing data...")

    batch_size = 128
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
    loss_weights[0] *= 0.05  # Analysis confidence
    loss_weights[1] *= 1  # CL
    loss_weights[2] *= 2  # ln(CD)
    loss_weights[3] *= 0.5  # CM
    loss_weights[4] *= 0.25  # Top Xtr
    loss_weights[5] *= 0.25  # Bot Xtr
    loss_weights[6:] *= 1 / (32 * 6)  # Lower the weight on all boundary layer outputs


    def loss_function(y_pred, y_data, return_individual_loss_components=False):
        # For data with NaN, overwrite the data with the prediction. This essentially makes the model ignore NaN data,
        # since the gradient of the loss with respect to parameters is zero when the data is NaN.
        y_data = torch.where(
            torch.isnan(y_data),
            y_pred,
            y_data
        )

        analysis_confidence_loss = torch.nn.functional.binary_cross_entropy(y_pred[:, 0], y_data[:, 0])
        # other_loss_components = torch.mean(
        #     (y_pred[:, 1:] - y_data[:, 1:]) ** 2,
        #     dim=0
        # )
        other_loss_components = torch.mean(
            torch.nn.functional.huber_loss(
                y_pred[:, 1:], y_data[:, 1:],
                reduction='none',
                delta=1
            ),
            dim=0
        )

        unweighted_loss_components = torch.stack([
            analysis_confidence_loss,
            *other_loss_components
        ], dim=0)
        weighted_loss_components = unweighted_loss_components * loss_weights
        loss = torch.sum(weighted_loss_components)

        if return_individual_loss_components:
            return weighted_loss_components
        else:
            return loss


    # raise Exception
    print(f"Training...")
    unweighted_epoch_loss_components = torch.ones(N_outputs, dtype=torch.float32).to(device)

    n_batches_per_epoch = len(train_loader)

    num_epochs = 100000000
    for epoch in range(num_epochs):
        # Put the model in training mode
        net.train()

        loss_from_each_training_batch = []

        for x, y_data in tqdm(train_loader):

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

        scheduler.step(train_loss)

        torch.save(net.state_dict(), cache_file)
