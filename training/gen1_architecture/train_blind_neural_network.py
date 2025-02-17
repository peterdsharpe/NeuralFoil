import aerosandbox.numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from training_data.load_data import df_train, df_test
import torch
from torch.utils.data import TensorDataset, DataLoader
import polars as pl


def convert_dataframe_to_inputs_outputs(df):
    df_inputs = pl.DataFrame({
        "4 * sin(2 * alpha)"       : 4 * np.sind(2 * df_train["alpha"]),
        "20 * (1 - cos^2(alpha))"  : 20 * (1 - np.cosd(df_train["alpha"]) ** 2),
        "(ln_Re - 12.5) / 2"       : (np.log(df_train["Re"]) - 12.5) / 2,
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
        "CL"              : df_train["CL"],
        "ln_CD + 4"       : np.log(df_train["CD"]) + 4,
        "CM * 20"         : df_train["CM"] * 20,
        "u_max_over_u - 1": (1 - df_train["Cpmin"]) ** 0.5,
        "Top_Xtr"         : df_train["Top_Xtr"],
        "Bot_Xtr"         : df_train["Bot_Xtr"],
    })

    return df_inputs, df_outputs


loss_weights = torch.tensor(list({
                                     "CL"          : 1,
                                     "ln_CD + 4"   : 1,
                                     "CM * 20"     : 0.25,
                                     "u_max_over_u": 0.25,
                                     "Top_Xtr"     : 0.25,
                                     "Bot_Xtr"     : 0.25,
                                 }.values())).reshape(1, -1)

df_train_inputs, df_train_outputs = convert_dataframe_to_inputs_outputs(df_train)
df_test_inputs, df_test_outputs = convert_dataframe_to_inputs_outputs(df_test)

cache_file = Path(__file__).parent / "nn-xxxlarge.pth"


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

    def forward(self, x: torch.Tensor):
        ### First, evaluate the network normally
        y = self.net(x)

        ### Then, flip the inputs and evaluate the network again.
        # The goal here is to embed the invariant of "symmetry across alpha" into the network evaluation.

        x_flipped = x.clone()
        x_flipped[:, 0] *= -1  # flip sin(alpha)
        x_flipped[:, 3:11] = x[:, 11:19] * -1  # Replace kulfan_lower with a flipped kulfan_upper
        x_flipped[:, 11:19] = x[:, 3:11] * -1  # Replace kulfan_upper with a flipped kulfan_lower
        x_flipped[:, 19] *= -1  # flip kulfan_LE_weight

        y_flipped = self.net(x_flipped)

        ### The resulting outputs will also be flipped, so we need to flip them back to their normal orientation
        y_flipped[:, 0] *= -1  # CL
        y_flipped[:, 2] *= -1  # CM
        temp = y_flipped[:,
               4].clone()  # This is just here to facilitate swapping the top / bottom Xtr (transition x) values
        y_flipped[:, 4] = y_flipped[:, 5]  # Replace top Xtr with bottom Xtr
        y_flipped[:, 5] = temp  # Replace bottom Xtr with top Xtr

        ### Then, average the two outputs to get the "symmetric" result
        return (y + y_flipped) / 2


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
    print("Preparing data...")

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
        num_workers=16,
    )

    test_inputs = torch.tensor(
        df_test_inputs.to_numpy(),
        dtype=torch.float32,
    )
    test_outputs = torch.tensor(
        df_test_outputs.to_numpy(),
        dtype=torch.float32,
    )
    test_loader = DataLoader(
        dataset=TensorDataset(test_inputs, test_outputs),
        batch_size=65536,
        num_workers=16,
    )

    # raise Exception
    print("Training...")

    n_batches_per_epoch = len(train_loader)
    loss_weights = loss_weights.to(device)

    num_epochs = 10000
    for epoch in range(num_epochs):

        # Train the model
        net.train()

        batch_losses = []

        for i, (x, y) in enumerate(train_loader):

            x = x.to(device)
            y = y.to(device)

            y_pred = net(x)

            loss = torch.mean(loss_weights * (y_pred - y) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())

        train_loss = np.mean(batch_losses)

        # Evaluate the model
        net.eval()

        batch_losses = []
        batch_residual_mae = []  # Residual mean absolute error (MAE, L1 norm) of each test batch

        for i, (x, y) in enumerate(test_loader):
            with torch.no_grad():
                x = x.to(device)
                y = y.to(device)

                y_pred = net(x)

                batch_losses.append(
                    torch.mean(loss_weights * (y_pred - y) ** 2).item()
                )
                batch_residual_mae.append(
                    torch.mean(torch.abs(y_pred - y), dim=0).cpu().numpy()
                )

        test_loss = np.mean(batch_losses)
        test_residual_mae = np.mean(np.stack(batch_residual_mae, axis=0), axis=0)

        labeled_maes = {
            "CL"          : test_residual_mae[0],
            "ln_CD"       : test_residual_mae[1],
            "CM"          : test_residual_mae[2] / 20,
            "u_max_over_u": test_residual_mae[3] * 2,
            "Top_Xtr"     : test_residual_mae[4],
            "Bot_Xtr"     : test_residual_mae[5],
        }
        print(
            f"Epoch: {epoch} | Train Loss: {train_loss.item():.6g} | Test Loss: {test_loss.item():.6g} | " + " | ".join(
                [
                    f"{k}: {v:.6g}"
                    for k, v in labeled_maes.items()
                ]))

        scheduler.step(train_loss)

        torch.save(net.state_dict(), cache_file)
