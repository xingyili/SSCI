import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from gcforest.gcforest import GCForest

nfolds = 5


def get_toy_config():
    config = {}
    ca_config = {}
    ca_config["random_state"] = 1
    ca_config["max_layers"] = 100
    ca_config["early_stopping_rounds"] = 3
    ca_config["n_classes"] = 2
    ca_config["estimators"] = [
        {"n_folds": 3, "type": "ExtraTreesClassifier", "n_estimators": 100, "max_depth": None, "n_jobs": -1,
         "max_features": 1},
        {"n_folds": 3, "type": "ExtraTreesClassifier", "n_estimators": 100, "max_depth": None, "n_jobs": -1,
         "max_features": 1},
        {"n_folds": 3, "type": "RandomForestClassifier", "n_estimators": 100, "max_depth": None, "n_jobs": -1},
        {"n_folds": 3, "type": "RandomForestClassifier", "n_estimators": 100, "max_depth": None, "n_jobs": -1}
    ]
    config["cascade"] = ca_config
    return config


if __name__ == "__main__":
    d = np.loadtxt("data/topology_features.tsv", delimiter='\t', skiprows=1, dtype=str)
    d = d[:, 1:].astype(np.float32)
    d_idx = np.arange(d.shape[0])
    p_idx = d_idx[d[:, -1] == 1]
    u_idx = d_idx[d[:, -1] == 0]

    p = d[d[:, -1] == 1, :]
    u = d[d[:, -1] == 0, :]
    x_p = p[:, 0:-1]
    y_p = p[:, -1]
    x_u = u[:, 0:-1]
    X_n = x_u[0]
    y_u = u[:, -1]

    X_n_index = np.array([])

    eRecalls = np.zeros(nfolds)
    ePrecisions = np.zeros(nfolds)
    ePRAUCs = np.zeros(nfolds)
    i = 0
    for i in range(nfolds):
        u_idx_u, u_idx_m = train_test_split(u_idx, test_size=0.2)
        x_u_u = d[:, 0:-1][u_idx_u]
        x_u_m = d[:, 0:-1][u_idx_m]
        y_u_u = d[:, -1][u_idx_u]
        y_u_m = d[:, -1][u_idx_m]

        x = np.concatenate((x_p, x_u_m), axis=0)
        y = np.concatenate((y_p, y_u_m), axis=0)

        scaler = StandardScaler().fit(x)
        
        x_train_transformed = scaler.transform(x)
        x_u_train_transformed = scaler.transform(x_u_u)

        config = get_toy_config()
        gc1 = GCForest(config)
        gc1.fit_transform(x_train_transformed, y)
        scores = gc1.predict_proba(x_u_train_transformed)[:, 1]
        orderScores = np.argsort(scores)
        orderList = [str(item) for item in orderScores]
        orderStr = ','.join(orderList)
        top = int(y_u.shape[0] * 0.1)
        topNIndex = orderScores[:top]
        t = 0
        while t < top:
            index = topNIndex[t]
            if u_idx_u[index] not in X_n_index:
                X_n_index = np.append(X_n_index, u_idx_u[index])
                x_n = x_u_u[index]
            t += 1
    
    X_u_index = np.array([])
    X_p_index = p_idx
    print(len(X_p_index))
    print(len(X_n_index))
    for i in range(len(d)):
        if i not in X_p_index and i not in X_n_index:
            X_u_index = np.append(X_u_index, i)
    print(len(X_u_index))

    joblib.dump(X_p_index, "joblib/p_idx.joblib")
    joblib.dump(X_n_index, "joblib/n_idx.joblib")
    joblib.dump(X_u_index, "joblib/u_idx.joblib")