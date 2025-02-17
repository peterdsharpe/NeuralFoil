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

    repeats = 10

    t = Timer()
    t.tic()

    for _ in range(repeats):
        outputs = get_aero_from_kulfan_parameters(
            kulfan_parameters={
                k: np.stack([af.kulfan_parameters[k] for af in afs], axis=-1)
                for k in afs[0].kulfan_parameters.keys()
            },
            alpha=alpha,
            Re=Re_f,
            model_size=model_size,
        )
    outputs["time"] = t.toc() / len(afs) / repeats

    df = pd.DataFrame(
        {
            "camber": camber_f,
            "thickness": thickness_f,
            "Re": Re_f,
            **{k: outputs[k] for k in ["CL", "CD", "CM", "time"]},
        }
    )

    df.to_csv(f"data/vect_nf_{model_size}.csv")
