import aerosandbox as asb
import aerosandbox.numpy as np
from pathlib import Path
import sys, os

sys.path.append(str(Path(__file__).parent.parent))
from data.load_data import df_train, df_test, weights, kulfan_cols, aero_input_cols, aero_output_cols, all_cols
import torch
from torch.utils.data import TensorDataset, DataLoader
import polars as pl


def convert_dataframe_to_inputs_outputs(df):
    df_inputs = pl.DataFrame({
        "alpha / 10"               : df_train["alpha"] / 10,
        "log10_Re - 5"             : np.log10(df_train["Re"]) - 5,
        'kulfan_lower_0 * 5'       : df_train['kulfan_lower_0'] * 5,
        'kulfan_lower_1 * 5'       : df_train['kulfan_lower_1'] * 5,
        'kulfan_lower_2 * 5'       : df_train['kulfan_lower_2'] * 5,
        'kulfan_lower_3 * 5'       : df_train['kulfan_lower_3'] * 5,
        'kulfan_lower_4 * 5'       : df_train['kulfan_lower_4'] * 5,
        'kulfan_lower_5 * 5'       : df_train['kulfan_lower_5'] * 5,
        'kulfan_lower_6 * 5'       : df_train['kulfan_lower_6'] * 5,
        'kulfan_lower_7 * 5'       : df_train['kulfan_lower_7'] * 5,
        'kulfan_upper_0 * 5'       : df_train['kulfan_upper_0'] * 5,
        'kulfan_upper_1 * 5'       : df_train['kulfan_upper_1'] * 5,
        'kulfan_upper_2 * 5'       : df_train['kulfan_upper_2'] * 5,
        'kulfan_upper_3 * 5'       : df_train['kulfan_upper_3'] * 5,
        'kulfan_upper_4 * 5'       : df_train['kulfan_upper_4'] * 5,
        'kulfan_upper_5 * 5'       : df_train['kulfan_upper_5'] * 5,
        'kulfan_upper_6 * 5'       : df_train['kulfan_upper_6'] * 5,
        'kulfan_upper_7 * 5'       : df_train['kulfan_upper_7'] * 5,
        'kulfan_LE_weight * 5'     : df_train['kulfan_LE_weight'] * 5,
        "kulfan_TE_thickness * 100": df_train["kulfan_TE_thickness"] * 100,
    })

    df_outputs = pl.DataFrame({
        "CL"          : df_train["CL"],
        "log10_CD + 2": np.log10(df_train["CD"]) + 2,
        "CM * 20"     : df_train["CM"] * 20,
        "Cpmin / 2"   : df_train["Cpmin"] / 2,
        "Top_Xtr"     : df_train["Top_Xtr"],
        "Bot_Xtr"     : df_train["Bot_Xtr"],
    })

    return df_inputs, df_outputs


df_train_inputs, df_train_outputs = convert_dataframe_to_inputs_outputs(df_train)
df_test_inputs, df_test_outputs = convert_dataframe_to_inputs_outputs(df_test)

cache_file = Path(__file__).parent / "nn-xxlarge.pth"

# Define the model
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()

        width = 512

        self.net = torch.nn.Sequential(
            torch.nn.Linear(len(df_train_inputs.columns), width),
            torch.nn.Tanh(),
            torch.nn.Linear(width, width),
            torch.nn.Tanh(),
            torch.nn.Linear(width, width),
            torch.nn.Tanh(),
            torch.nn.Linear(width, width),
            torch.nn.Tanh(),
            torch.nn.Linear(width, len(df_train_outputs.columns)),
        )

    def forward(self, x):
        return self.net(x)


if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    net = Net().to(device)

    # try:
    #     net.load_state_dict(torch.load(cache_file))
    #     print("Model found, resuming training.")
    # except FileNotFoundError:
    #     print("No existing model found, starting fresh.")

    # Define the optimizer
    learning_rate = 1e-4
    optimizer = torch.optim.RAdam(net.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=20,
        verbose=True,
        min_lr=1e-6,
    )

    # Define the data loader
    print(f"Preparing data...")

    batch_size = 512
    train_inputs = torch.tensor(
        df_train_inputs.to_numpy(),
        dtype=torch.float32,
    )
    train_outputs = torch.tensor(
        df_train_outputs.to_numpy(),
        dtype=torch.float32,
    )
    train_loader = DataLoader(
        dataset=TensorDataset(train_inputs, train_outputs),
        batch_size=batch_size,
        shuffle=True,
        num_workers=20,
    )

    test_inputs = torch.tensor(
        df_test_inputs.to_numpy(),
        dtype=torch.float32,
        device=device,
    )
    test_outputs = torch.tensor(
        df_test_outputs.to_numpy(),
        dtype=torch.float32,
        device=device,
    )

    # raise Exception
    print(f"Training...")

    n_batches_per_epoch = len(train_loader)

    # Train the model
    num_epochs = 10000
    for epoch in range(num_epochs):
        net.train()

        batch_losses = []

        for i, (x, y) in enumerate(train_loader):

            x = x.to(device)
            y = y.to(device)

            y_pred = net(x)

            loss = torch.mean((y_pred - y) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())

        train_loss = np.mean(batch_losses)

        # Evaluate the model
        net.eval()
        with torch.no_grad():
            y_test_pred = net(test_inputs)
            test_residuals = y_test_pred - test_outputs
            test_loss = torch.mean(test_residuals ** 2)
            errors = {
                "CL"     : test_residuals[:, 0],
                "CD"     : 10 ** (y_test_pred[:, 1] - 2) - 10 ** (test_outputs[:, 1] - 2),
                "CM"     : test_residuals[:, 2] / 20,
                "Cpmin"  : test_residuals[:, 3] * 2,
                "Top_Xtr": test_residuals[:, 4],
                "Bot_Xtr": test_residuals[:, 5],
            }
        print(
            f"Epoch: {epoch} | Train Loss: {train_loss.item():.6g} | Test Loss: {test_loss.item():.6g} | " + " | ".join(
                [
                    f"{k}: {v.abs().median():.6g}"
                    for k, v in errors.items()
                ]))

        scheduler.step(train_loss)

        torch.save(net.state_dict(), cache_file)
