import numpy as np, pandas as pd
cap = "../capsule-2202829/"
cv = cap + "code/crossvalidation/"
for f in ["X.npy", "Y.npy", "crosslabel.npy", "importanceidx.npy"]:
    a = np.load(cv + f, allow_pickle=True); print(f, a.shape, a.dtype)
for n in ["X_aux", "X_unnormed"]:
    a = np.load("data/raw/%s.npy" % n); b = np.load(cap + "code/src/model/%s.npy" % n)
    print(n, "data/raw == capsule:", a.shape == b.shape and np.array_equal(a, b))
sup = pd.read_csv("data/labels/ab_bind_s645_supplement.csv", keep_default_na=False)
y = sup.ddg_abbind.values
Y = np.load(cv + "Y.npy", allow_pickle=True).ravel()
print("len Y:", len(Y))
if len(Y) == 645:
    print("same order  (Y == ddg_abbind):", np.allclose(Y, y))
    print("flipped sign:", np.allclose(Y, -y))
    print("same values, different order:", np.allclose(np.sort(Y), np.sort(y)) and not np.allclose(Y, y))
ab = pd.read_csv(cap + "data/AB-Bind-Database/AB_645.csv")
print(ab.shape); print(ab.head(3).to_string())
