from make_cases import *

outputs = []

for i in tqdm(range(len(afs))):
    xf = asb.XFoil(
        airfoil=afs[i],
        Re=Re_f[i],
        mach=0,
        verbose=False,
        xfoil_repanel=True,
        max_iter=100,
    )
    t = Timer()
    t.tic()
    out = xf.alpha(alpha)
    out["time"] = t.toc()
    outputs.append(out)

df = pd.DataFrame({
    "camber"   : camber_f,
    "thickness": thickness_f,
    "Re"       : Re_f,
    **{
        k: np.concatenate([
            np.atleast_1d(outputs[i][k]) if len(np.atleast_1d(outputs[i][k])) != 0 else np.array([np.nan])
            for i in range(len(afs))
        ])
        for k in outputs[0].keys()
    }
})

df.to_csv("data/reference_data.csv")