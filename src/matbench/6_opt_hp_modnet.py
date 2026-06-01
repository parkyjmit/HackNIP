# script to optimize modnet hyper-parameters (nested CV)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   

import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("PyTorch device:", device)
print("TensorFlow GPUs:", tf.config.list_logical_devices("GPU"))

import random, pathlib, pickle, optuna 
import numpy as np                                  
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from modnet.preprocessing import MODData
from modnet.models import MODNetModel
from run_benchmark import parse_task_list
from pathlib import Path

DATA_ROOT = pathlib.Path(os.environ.get("BENCH_DATA_DIR", pathlib.Path(__file__).resolve().parent / "benchmark_data")).resolve()
MLIP      = os.environ.get("BENCH_MLIP", "orb2")
MODEL     = os.environ.get("BENCH_MODEL", "modnet")
TASKS     = os.environ.get("BENCH_TASKS")

STRUCTURES_DIR = DATA_ROOT / "structures"
META_DIR       = DATA_ROOT / "metadata"
FEAT_DIR       = DATA_ROOT / f"feat_{MLIP}"
NPY_DIR        = FEAT_DIR / "npy"
RESULTS_DIR    = FEAT_DIR / f"results_{MODEL}"
HP_DIR         = RESULTS_DIR / "hp"
PARITY_DIR     = RESULTS_DIR / "parity"

for p in [STRUCTURES_DIR, META_DIR, FEAT_DIR, NPY_DIR, RESULTS_DIR, HP_DIR, PARITY_DIR]:
    p.mkdir(parents=True, exist_ok=True)

KEY        = "XPS"
key        = KEY.lower()
L_MIN, L_MAX = 1, 15  # valid layer bounds
n_trials   = 50       # total hyperparameter evaluations

# CV splitters
matbench_seed = 18012019
outer_cv = KFold(n_splits=5, shuffle=True, random_state=matbench_seed)
inner_cv = KFold(n_splits=5, shuffle=True, random_state=matbench_seed)

def make_early_stop_callback(patience=10, atol=0.0):
    state = {"best": float("inf"), "stale": 0}
    def _cb(study, trial):
        val = study.best_value
        if val + atol < state["best"]:
            state["best"] = val
            state["stale"] = 0
        else:
            state["stale"] += 1
            if state["stale"] >= patience:
                print(f"[EarlyStop] No improvement for {state['stale']} trials → stopping.")
                study.stop()
    return _cb


def build_modnet_model(N_features, blocks, out_act):
    return MODNetModel(
        targets=[["g"]],
        weights={"g": 1.0},
        num_neurons=blocks,
        n_feat=N_features,
        num_classes={"g": 0},
        out_act=out_act
    )

def objective_inner_cv(trial, X_train_outer, y_train_outer):
    # 1) suggest hyperparameters
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 128])
    lr         = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)
    N_features = trial.suggest_int("N_features", 50, 256)
    depth      = trial.suggest_int("depth", 1, 4)
    width      = trial.suggest_int("width", 32, 256, log=True)
    hidden = [[width] for _ in range(depth)]
    blocks = tuple(hidden) + ([],) * (4 - depth)

    loss    = trial.suggest_categorical("loss", ["mae"])
    out_act = trial.suggest_categorical("out_act", ["linear"])

    maes = []

    # 2) INNER CV loop 
    for tr_idx, va_idx in inner_cv.split(X_train_outer):
        X_tr, X_va = X_train_outer[tr_idx], X_train_outer[va_idx]
        y_tr, y_va = y_train_outer[tr_idx], y_train_outer[va_idx]

        # select top-N features
        cols = list(pd.DataFrame(X_tr).columns)[:N_features]

        # wrap into MODData
        md_tr = MODData(
            df_featurized=pd.DataFrame(X_tr[:, :N_features], columns=cols),
            targets=pd.Series(y_tr),
            target_names=["g"]
        )
        md_tr.optimal_features = cols

        md_va = MODData(
            df_featurized=pd.DataFrame(X_va[:, :N_features], columns=cols),
            targets=pd.Series(y_va),
            target_names=["g"]
        )
        md_va.optimal_features = cols

        # build & fit the model
        model = build_modnet_model(N_features=N_features, blocks=blocks, out_act=out_act)
        model.fit(md_tr, batch_size=batch_size, lr=lr, loss=loss)

        y_pred = model.predict(md_va, remap_out_of_bounds=False).squeeze()
        maes.append(mean_absolute_error(y_va, y_pred))

    mean_mae = float(np.mean(maes))
    std_mae  = float(np.std(maes))
    trial.set_user_attr("mae_std", std_mae)
    return mean_mae


# after tuning, fit on FULL outer-train and evaluate on outer-test
def fit_and_eval_on_outer_test(best_params, X_train_outer, y_train_outer, X_test_outer, y_test_outer):
    batch_size = best_params["batch_size"]
    lr         = best_params["learning_rate"]
    N_features = best_params["N_features"]

    depth = best_params["depth"]
    width = best_params["width"]
    hidden = [[width] for _ in range(depth)]
    blocks = tuple(hidden) + ([],) * (4 - depth)

    loss    = best_params["loss"]
    out_act = best_params["out_act"]

    cols = list(pd.DataFrame(X_train_outer).columns)[:N_features]

    md_tr_full = MODData(
        df_featurized=pd.DataFrame(X_train_outer[:, :N_features], columns=cols),
        targets=pd.Series(y_train_outer),
        target_names=["g"]
    )
    md_tr_full.optimal_features = cols

    md_te = MODData(
        df_featurized=pd.DataFrame(X_test_outer[:, :N_features], columns=cols),
        targets=pd.Series(y_test_outer),
        target_names=["g"]
    )
    md_te.optimal_features = cols

    model = build_modnet_model(N_features=N_features, blocks=blocks, out_act=out_act)
    model.fit(md_tr_full, batch_size=batch_size, lr=lr, loss=loss)

    y_pred = model.predict(md_te, remap_out_of_bounds=False).squeeze()
    return float(mean_absolute_error(y_test_outer, y_pred))

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("PyTorch device:", device)
    print("TensorFlow GPUs:", tf.config.list_logical_devices("GPU"))

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    task_slugs = parse_task_list(TASKS)

    for task in task_slugs[:]:
        feat = pickle.load(open(os.path.join(FEAT_DIR, f'{task}_{KEY}_{MLIP}.pkl'),'rb'))
        score_path = pathlib.Path(RESULTS_DIR) / "benchmark_scores.csv"
        score_df = pd.read_csv(score_path)

        row = score_df.loc[score_df["task"] == task]
        base_layer = int(row.iloc[0]["layer"])

        candidate_layers = list(range(
            max(L_MIN, base_layer - 2),
            min(L_MAX, base_layer + 2) + 1
        ))

        # master record for this task
        master_csv = os.path.join(HP_DIR, f"t{task}_{KEY}_{MLIP}_optuna.csv")
        first = not os.path.exists(master_csv)

        best_tasks = []

        for l in candidate_layers:
            X_all = feat[f"{KEY}_l{l}"]
            y_all = feat["targets"]

            # outer-fold evaluation list
            outer_test_maes = []
            outer_fold_records = []

            # nested CV outer loop
            for fold_id, (train_idx, test_idx) in enumerate(outer_cv.split(X_all), start=1):
                X_train_outer, X_test_outer = X_all[train_idx], X_all[test_idx]
                y_train_outer, y_test_outer = y_all[train_idx], y_all[test_idx]

                cb = make_early_stop_callback(patience=10)
                study = optuna.create_study(
                    direction="minimize",
                    sampler=optuna.samplers.TPESampler(seed=seed)
                )

                # Optuna objective sees ONLY outer-train data
                study.optimize(
                    lambda t: objective_inner_cv(t, X_train_outer, y_train_outer),
                    n_trials=n_trials,
                    n_jobs=2,
                    callbacks=[cb],
                    show_progress_bar=True
                )

                # dump every trial if you'd like
                study.trials_dataframe().to_csv(
                    os.path.join(HP_DIR, f"t{task}_l{l}_outerfold{fold_id}_optuna.csv"),
                    index=False
                )

                best = study.best_trial

                # fit on full outer-train, evaluate on outer-test (unseen during tuning)
                outer_mae = fit_and_eval_on_outer_test(
                    best_params=best.params,
                    X_train_outer=X_train_outer,
                    y_train_outer=y_train_outer,
                    X_test_outer=X_test_outer,
                    y_test_outer=y_test_outer
                )
                outer_test_maes.append(outer_mae)

                outer_fold_records.append({
                    "task": task,
                    "layer": l,
                    "outer_fold": fold_id,
                    "innercv_best_mae": float(best.value),
                    "innercv_best_mae_std": float(best.user_attrs.get("mae_std", np.nan)),
                    "outer_test_mae": float(outer_mae),
                    "trial": int(best.number),
                    **best.params
                })

                print(f"[task {task} l{l} fold {fold_id}] "
                      f"best inner-CV MAE={best.value:.4f} → outer-test MAE={outer_mae:.4f}")

            mean_outer_mae = float(np.mean(outer_test_maes))
            std_outer_mae  = float(np.std(outer_test_maes))

            per_fold_csv = os.path.join(HP_DIR, f"t{task}_l{l}_folds_nest.csv")
            pd.DataFrame(outer_fold_records).to_csv(per_fold_csv, index=False)

            rec = {
                "task": task,
                "layer": l,
                "best_mae": mean_outer_mae,
                "best_mae_std": std_outer_mae,
                "innercv_best_mae_mean": float(np.mean([r["innercv_best_mae"] for r in outer_fold_records])),
            }

            pd.DataFrame([rec]).to_csv(master_csv, mode="a", header=first, index=False)
            first = False

            best_tasks.append(rec)
            print(f"[task {task} l{l}] OUTER mean MAE = {mean_outer_mae:.4f} ± {std_outer_mae:.4f}")

        # after all layers, record the best-layer for this task
        best_overall = min(best_tasks, key=lambda r: r["best_mae"])
        task_csv = os.path.join(HP_DIR, "benchmark_optuna_nest.csv")
        pd.DataFrame([best_overall]).to_csv(
            task_csv,
            mode="a",
            header=not os.path.exists(task_csv),
            index=False
        )


if __name__ == "__main__":
    main()
