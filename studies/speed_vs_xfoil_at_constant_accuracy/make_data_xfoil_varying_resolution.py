from make_cases import *

for N in [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260]:
    print(f"N = {N}")

    outputs = []

    for i in tqdm(range(len(afs))):
        xf = asb.XFoil(
            airfoil=afs[i],
            Re=Re_f[i],
            mach=0,
            verbose=False,
            xfoil_repanel=True,
            xfoil_repanel_n_points=N,
            max_iter=100,
        )
        t = Timer()
        t.tic()
        out = xf.alpha(alpha)
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
                for k in outputs[0].keys()
            },
        }
    )

    df.to_csv(f"data/xfoil_{N}.csv")
