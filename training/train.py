from pathlib import Path
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

from export import export_checkpoint
from model import Net
from neuralfoil import _spec

N_inputs = len(df_train_inputs_scaled.columns)
N_outputs = len(df_train_outputs_scaled.columns)

cache_file = Path(__file__).parent / "nn-xxxlarge.pth"
n_hidden_layers = 5  # Counts only the width-to-width blocks; the total number of hidden neuron layers is n_hidden_layers + 1 (due to the input-to-width layer).
width = 512
print("Cache file: ", cache_file)


if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)

    net = Net(
        mean_inputs_scaled=torch.tensor(mean_inputs_scaled, dtype=torch.float32),
        cov_inputs_scaled=torch.tensor(cov_inputs_scaled, dtype=torch.float32),
        n_hidden_layers=n_hidden_layers,
        width=width,
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
    print("Preparing data...")

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
        y_data = torch.where(torch.isnan(y_data), y_pred, y_data)

        analysis_confidence_loss = torch.mean(
            torch.nn.functional.binary_cross_entropy_with_logits(
                input=y_pred[:, 0:1],
                target=y_data[:, 0:1],
                reduction="none",
            ),
            dim=0,
        )
        # other_loss_components = torch.mean(
        #     (y_pred[:, 1:] - y_data[:, 1:]) ** 2,
        #     dim=0
        # )

        other_loss_components = torch.mean(
            torch.nn.functional.huber_loss(
                y_pred[:, 1:], y_data[:, 1:], reduction="none", delta=0.05
            ),
            dim=0,
        )

        # other_loss_components = torch.mean(
        #     torch.nn.functional.mse_loss(
        #         y_pred[:, 1:], y_data[:, 1:],
        #         reduction='none',
        #     ),
        #     dim=0
        # )

        unweighted_loss_components = torch.concatenate(
            [analysis_confidence_loss, other_loss_components], dim=0
        )

        weighted_loss_components = unweighted_loss_components * loss_weights

        if return_individual_loss_components:
            return weighted_loss_components
        else:
            return torch.sum(weighted_loss_components)

    # raise Exception
    print("Training...")

    n_batches_per_epoch = len(train_loader)

    num_epochs = 10**9  # Effectively loop until manually stopped
    for epoch in range(num_epochs):
        # Put the model in training mode
        net.train()

        loss_from_each_training_batch = []

        # for x, y_data in tqdm(train_loader):
        for x, y_data in train_loader:

            x = x.to(device)
            y_data = y_data.to(device)

            loss = loss_function(y_pred=net(x), y_data=y_data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_from_each_training_batch.append(loss.detach())

        train_loss = torch.mean(
            torch.stack(loss_from_each_training_batch, dim=0), dim=0
        )

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
                    y_pred=y_pred, y_data=y_data, return_individual_loss_components=True
                )

                loss_components_from_each_test_batch.append(loss_components)

                y_pred[:, 0] = torch.sigmoid(
                    y_pred[:, 0]
                )  # Analysis confidence, a binary variable

                mae_from_each_test_batch.append(
                    torch.nanmean(torch.abs(y_pred - y_data), dim=0)
                )

        test_loss_components = torch.mean(
            torch.stack(loss_components_from_each_test_batch, dim=0), dim=0
        )
        test_loss = torch.sum(test_loss_components)
        test_residual_mae = torch.nanmean(
            torch.stack(mae_from_each_test_batch, dim=0), dim=0
        )

        labeled_maes = {
            "analysis_confidence": test_residual_mae[0],
            "CL": test_residual_mae[1] / _spec.CL_SCALE,
            "ln_CD": test_residual_mae[2] * _spec.LN_CD_SCALE,
            "CM": test_residual_mae[3] / _spec.CM_SCALE,
            "Top_Xtr": test_residual_mae[4],
            "Bot_Xtr": test_residual_mae[5],
        }
        print(
            f"Epoch: {epoch} | Train Loss: {train_loss.item():.6g} | Test Loss: {test_loss.item():.6g} | "
            + " | ".join([f"{k}: {v:.6g}" for k, v in labeled_maes.items()])
        )
        loss_argsort = torch.argsort(test_loss_components, descending=True)
        print("Loss contributors: ")
        for i in loss_argsort[:10]:
            print(
                f"\t{df_train_outputs_scaled.columns[i]:25}: {test_loss_components[i].item():.6g}"
            )

        scheduler.step(test_loss)

        torch.save(
            {
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            cache_file,
        )
        # Keep a current release artifact next to the checkpoint, so no manual
        # conversion step exists. Promoting it into the package is deliberate:
        # `python export.py <checkpoint> --install` (see export.py docstring).
        export_checkpoint(cache_file, cache_file.with_suffix(".npz"))
