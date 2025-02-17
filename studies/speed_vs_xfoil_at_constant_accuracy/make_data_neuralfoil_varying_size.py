from make_cases import *

from neuralfoil import get_aero_from_kulfan_parameters

for model_size in [
    "xxsmall",
    "xsmall",
    "small",
    "medium",
    "large",
    "xlarge",
    "xxlarge",
    "xxxlarge",
]:
    print(f"model_size = {model_size}")

    outputs = []

    for i in tqdm(range(len(afs))):

        t = Timer()
        t.tic()
        out = get_aero_from_kulfan_parameters(
            kulfan_parameters=afs[i].kulfan_parameters,
            alpha=alpha,
            Re=Re_f[i],
            model_size=model_size,
        )
        out["time"] = t.toc()
        outputs.append(out)

    df = pd.DataFrame(
        {
            "camber": camber_f,
            "thickness": thickness_f,
            "Re": Re_f,
            **{
                k: np.concatenate(
                    [
                        (
                            np.atleast_1d(outputs[i][k])
                            if len(np.atleast_1d(outputs[i][k])) != 0
                            else np.array([np.nan])
                        )
                        for i in range(len(afs))
                    ]
                )
                for k in ["CL", "CD", "CM", "time"]
            },
        }
    )

    df.to_csv(f"data/nf_{model_size}.csv")
