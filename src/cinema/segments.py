# сегментация зрителей по поведению в первые недели: признаки, kmeans, профили, сверка с заложенными типами

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from . import marts

FEATURES = ["log_hours", "hours_per_session", "titles_per_session", "weekend_share", "hours_cv"]
FEATURE_NAMES = {"log_hours": "часов в неделю (лог)", "hours_per_session": "часов за сессию",
                 "titles_per_session": "названий за сессию", "weekend_share": "доля выходных",
                 "hours_cv": "разброс по неделям"}


def build_features(conn):
    f = marts.user_features(conn)
    f = f[f["sum_sessions"] > 0].copy()
    f["log_hours"] = np.log(f["mean_hours"].clip(lower=0.1))
    f["hours_per_session"] = f["sum_hours"] / f["sum_sessions"]
    f["titles_per_session"] = f["sum_titles"] / f["sum_sessions"]
    f["hours_cv"] = np.sqrt(f["var_hours"].clip(lower=0)) / f["mean_hours"].clip(lower=0.1)
    # редкие выбросы по длине сессии режем, чтобы они не тянули центры кластеров
    for c in ["hours_per_session", "titles_per_session", "hours_cv"]:
        f[c] = f[c].clip(upper=f[c].quantile(0.995))
    return f.reset_index(drop=True)


def scale(f):
    sc = StandardScaler()
    return sc.fit_transform(f[FEATURES].values), sc


def scan_k(X, ks=range(2, 9), seed=7, sample=6000):
    # на каждом k считаем инерцию и силуэт, силуэт на подвыборке, потому что на всех точках долго
    rows = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
        sil = silhouette_score(X, km.labels_, sample_size=min(sample, len(X)), random_state=seed)
        rows.append({"k": k, "inertia": float(km.inertia_), "silhouette": float(sil)})
    return pd.DataFrame(rows)


def fit(X, k, seed=7):
    km = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(X)
    return km.labels_, km


def order_by_hours(f, labels):
    # нумеруем сегменты так, чтобы первый был самым активным, тогда номера не прыгают от запуска к запуску
    mean_h = pd.Series(f["mean_hours"].values).groupby(labels).mean().sort_values(ascending=False)
    mapping = {old: new for new, old in enumerate(mean_h.index)}
    return np.array([mapping[x] for x in labels])


def name_segments(profile):
    # грубые подписи по профилю, работают для четырех сегментов, иначе остаются номера
    k = len(profile)
    names = {i: f"S{i + 1}" for i in range(k)}
    if k != 4:
        return names
    left = list(profile.index)
    pick = {}
    i = profile.loc[left, "weekend_share"].idxmax()
    pick[i] = "выходные"
    left.remove(i)
    i = profile.loc[left, "hours_per_session"].idxmax()
    pick[i] = "марафонцы"
    left.remove(i)
    i = profile.loc[left, "mean_hours"].idxmin()
    pick[i] = "редкие зрители"
    left.remove(i)
    for i in left:
        pick[i] = "обычные"
    return {i: f"S{i + 1} {pick[i]}" for i in range(k)}


def profile(f, labels):
    d = f.copy()
    d["segment"] = labels
    g = d.groupby("segment")
    p = g[["mean_hours", "hours_per_session", "titles_per_session", "weekend_share", "hours_cv"]].mean()
    p["share"] = g.size() / len(d)
    p["users"] = g.size()
    return p


def agreement(labels, persona):
    tab = pd.crosstab(pd.Series(labels, name="segment"), pd.Series(persona, name="persona"))
    # чистота: какая доля людей в сегменте принадлежит самому частому типу
    purity = float(tab.max(axis=1).sum() / tab.values.sum())
    return {"ari": float(adjusted_rand_score(persona, labels)),
            "nmi": float(normalized_mutual_info_score(persona, labels)),
            "purity": purity, "table": tab}


def pca_points(X, labels, persona, n=7000, seed=7):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n, len(X)), replace=False)
    pca = PCA(n_components=2, random_state=seed).fit(X)
    xy = pca.transform(X[idx])
    return pd.DataFrame({"pc1": xy[:, 0], "pc2": xy[:, 1], "segment": labels[idx], "persona": persona[idx]}), \
        pca.explained_variance_ratio_


def run(conn, truth_users, k=4):
    f = build_features(conn)
    f = f.merge(truth_users[["user_id", "persona"]], on="user_id", how="left")
    X, _ = scale(f)
    scan = scan_k(X)
    labels, _ = fit(X, k)
    labels = order_by_hours(f, labels)
    prof = profile(f, labels)
    names = name_segments(prof)
    ag = agreement(labels, f["persona"].values)
    pts, evr = pca_points(X, labels, f["persona"].values)
    f["segment"] = labels
    return {"features": f, "scan": scan, "profile": prof, "names": names, "agreement": ag,
            "pca": pts, "pca_var": evr, "k": k}


def segment_survival(ep, f):
    # дожитие по сегментам, только месячные подписки и только те, кого мы сегментировали
    from .survival import kaplan_meier, km_at, km_median, logrank
    d = ep[ep["billing"] == "monthly"].merge(f[["user_id", "segment"]], on="user_id")
    km = {int(s): kaplan_meier(g["weeks"], g["event_voluntary"]) for s, g in d.groupby("segment")}
    rows = [{"segment": s, "n": int((d["segment"] == s).sum()), "median_weeks": km_median(k),
             "surv_26w": km_at(k, 26)} for s, k in km.items()]
    return km, pd.DataFrame(rows), logrank(d["weeks"], d["event_voluntary"], d["segment"])
