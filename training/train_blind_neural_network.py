import aerosandbox as asb
import aerosandbox.numpy as np
from pathlib import Path
import sys, os

sys.path.append(str(Path(__file__).parent.parent))
from data.load_data import df_train, df_test, weights, kulfan_cols, aero_input_cols, aero_output_cols, all_cols
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import polars as pl

cache_file = Path(__file__).parent / "blind_neural_network.pth"

df_inputs = df_train[aero_input_cols + kulfan_cols].with_columns(
    (pl.col("alpha") / 10).alias("alpha / 10"),
    (np.log10(df_train["Re"])).alias("log10_Re"),
).drop("alpha", "Re")

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


# Define the model
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(len(df_inputs.columns), 64),
            torch.nn.Tanh(),
            torch.nn.BatchNorm1d(64), # Add batch normalization after linear layers
            torch.nn.Dropout(0.2), # Add dropout after activation layers; rate can be adjusted
            torch.nn.Linear(64, 64),
            torch.nn.Tanh(),
            torch.nn.BatchNorm1d(64),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(64, 64),
            torch.nn.Tanh(),
            torch.nn.BatchNorm1d(64),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(64, len(df_outputs.columns)),
        )

    def forward(self, x):
        return self.net(x)


if __name__ == '__main__':
    # device = torch.device('cpu')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    net = Net()
    try:
        net.load_state_dict(torch.load(cache_file))
        print("Model found, resuming training.")
    except FileNotFoundError:
        print("No existing model found, starting fresh.")

    net = net.to(device)

    # Define the optimizer
    learning_rate = 1e-5
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate)

    # Define the data loader
    print(f"Preparing data...")

    batch_size = 64
    input_tensor = torch.tensor(
        df_inputs.to_numpy(),
        dtype=torch.float32,
        device=device,
    )
    output_tensor = torch.tensor(
        df_outputs.to_numpy(),
        dtype=torch.float32,
        device=device,
    )
    dataset = TensorDataset(input_tensor, output_tensor)
    train_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    # raise Exception
    print(f"Training...")
    net.train()

    # Train the model
    num_epochs = 10000
    for epoch in range(num_epochs):

        for i, (x, y) in enumerate(train_loader):

            # x = x.to(device)
            # y = y.to(device)

            # Forward pass
            y_pred = net(x)

            # Compute loss
            loss = torch.mean((y_pred - y) ** 2)
            # loss = torch.nn.HuberLoss()(y_pred, y)

            if i % 1000 == 0:
                residuals = y_pred - y
                errors = {
                    "CL"     : residuals[:, 0],
                    "CD"     : 10 ** (y_pred[:, 1] - 2) - 10 ** (y[:, 1] - 2),
                    "CM"     : residuals[:, 2] / 20,
                    "Cpmin"  : residuals[:, 3] * 2,
                    "Top_Xtr": residuals[:, 4],
                    "Bot_Xtr": residuals[:, 5],
                }
                print(f"Epoch: {epoch} | Batch: {i} | Loss: {loss.item():.8g} | " + " | ".join([
                    f"{k}: {v.abs().mean():.8g}"
                    for k, v in errors.items()
                ]))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        torch.save(net.state_dict(), cache_file)
