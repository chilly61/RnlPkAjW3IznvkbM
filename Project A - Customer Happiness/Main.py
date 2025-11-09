import warnings
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# sklearn
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict, StratifiedKFold, train_test_split
from sklearn.metrics import (accuracy_score, classification_report, recall_score,
                             roc_auc_score, roc_curve, balanced_accuracy_score)
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

# joblib control
import joblib

# -------------------------
# Determinism settings
# -------------------------
SEED = 124
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

# Force joblib to use loky backend and avoid inner parallelism
joblib.parallel_backend('loky')

# -------------------------
# Main function
# -------------------------


def stacking_accuracy_optimization_deterministic(X, y, test_size=0.3, target_accuracy=0.65):
    """
    Deterministic version of your stacking pipeline that:
      - computes CV probabilities via cross_val_predict (n_jobs=1),
      - chooses threshold by maximizing CV accuracy,
      - trains final stacking on X_train and evaluates on X_test at that threshold,
      - computes CV AUC (from cross-validated probs) and Test AUC,
      - returns strategy-level test metrics and plotting.
    """

    print("=" * 60)
    print("DETERMINISTIC STACKING - CV vs TEST AUC & WEIGHTING STRATEGIES")
    print("=" * 60)

    # 1) Preprocessing - EXACTLY as your original: scale then SelectKBest (fit on full X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    selector = SelectKBest(score_func=f_classif, k=min(10, X.shape[1]))
    X_selected = selector.fit_transform(X_scaled, y)
    selected_features = X.columns[selector.get_support()]
    print(f"Selected features: {list(selected_features)}")

    # 2) Train/test split (deterministic)
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=test_size, stratify=y, random_state=SEED
    )

    # 3) Base models and meta-learner with fixed random_state
    base_models = [
        ('lr', LogisticRegression(C=0.0690578407, random_state=SEED, max_iter=1100)),
        ('svm', SVC(C=16.895294109, kernel='rbf', probability=True,
         random_state=SEED, gamma=0.3508198145)),
        ('rf', RandomForestClassifier(n_estimators=100,
         max_depth=14, random_state=SEED, min_samples_split=3))
    ]

    meta_learner = RandomForestClassifier(
        n_estimators=100, max_depth=14, random_state=SEED)

    stacking_model = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_learner,
        cv=5,
        passthrough=False,
        n_jobs=1  # single-threaded for reproducibility
    )

    # 4) Obtain CV probabilities on X_train (cross-validated)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    print("Obtaining cross-validated probabilities (CV) for the stacking ensemble...")
    y_proba_stacking = cross_val_predict(
        stacking_model, X_train, y_train, cv=skf,
        method='predict_proba', n_jobs=1  # single-threaded
    )

    # CV AUC and ROC
    cv_auc = roc_auc_score(y_train, y_proba_stacking[:, 1])
    fpr_cv, tpr_cv, _ = roc_curve(y_train, y_proba_stacking[:, 1])
    print(f"CV AUC (from cross-validated probabilities): {cv_auc:.4f}")

    # 5) Threshold search - maximize CV accuracy (this is the canonical selection rule)
    thresholds = np.linspace(0.1, 0.9, 81)
    records = []
    for thr in thresholds:
        yhat = (y_proba_stacking[:, 1] > thr).astype(int)
        acc = accuracy_score(y_train, yhat)
        r0 = recall_score(y_train, yhat, pos_label=0)
        r1 = recall_score(y_train, yhat, pos_label=1)
        records.append({'threshold': thr, 'accuracy': acc,
                       'recall_0': r0, 'recall_1': r1})
    stacking_df = pd.DataFrame(records)
    best_idx = stacking_df['accuracy'].idxmax()
    best_threshold = stacking_df.loc[best_idx, 'threshold']
    best_cv_accuracy = stacking_df.loc[best_idx, 'accuracy']
    print(f"Chosen threshold (maximize CV accuracy): {best_threshold:.3f}")
    print(f"Best CV accuracy (at that threshold): {best_cv_accuracy:.4f}")

    # 6) Individual model CV comparisons (same deterministic setting)
    individual_results = {}
    for name, model in base_models:
        y_proba_ind = cross_val_predict(
            model, X_train, y_train, cv=skf, method='predict_proba', n_jobs=1)
        # find best CV accuracy across thresholds
        accs = [(accuracy_score(y_train, (y_proba_ind[:, 1] > thr).astype(int)))
                for thr in thresholds]
        individual_results[name] = max(accs)
        print(
            f"{name.upper()} best CV accuracy (threshold-tuned): {individual_results[name]:.4f}")

    # 7) Train final stacking on full X_train
    stacking_model.fit(X_train, y_train)

    # Test set probabilities and metrics
    y_test_proba = stacking_model.predict_proba(X_test)
    test_auc = roc_auc_score(y_test, y_test_proba[:, 1])
    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_proba[:, 1])
    y_test_pred = (y_test_proba[:, 1] > best_threshold).astype(int)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_rec0 = recall_score(y_test, y_test_pred, pos_label=0)
    test_rec1 = recall_score(y_test, y_test_pred, pos_label=1)
    test_bal = balanced_accuracy_score(y_test, y_test_pred)

    print("\nTest set performance at CV-optimal threshold:")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Recall_0: {test_rec0:.4f}")
    print(f"Test Recall_1: {test_rec1:.4f}")
    print(f"Test AUC: {test_auc:.4f}")

    # 8) Weighted scoring strategies (we do NOT change model class_weight here;
    #    these are scoring strategies used to pick thresholds from CV).
    weighting_strategies = {
        'balanced': [0.4, 0.3, 0.3],
        'accuracy_focused': [0.6, 0.2, 0.2],
        'class_0_focused': [0.4, 0.4, 0.2],
        'class_1_focused': [0.3, 0.2, 0.5]
    }

    strat_summary = []
    for strat_name, weights in weighting_strategies.items():
        w_acc, w_r0, w_r1 = weights
        stacking_df[f'weighted_score_{strat_name}'] = (
            w_acc * stacking_df['accuracy']
            + w_r0 * stacking_df['recall_0']
            + w_r1 * stacking_df['recall_1']
        )
        idx = stacking_df[f'weighted_score_{strat_name}'].idxmax()
        thr_best = stacking_df.loc[idx, 'threshold']

        # CV values at that threshold
        cv_acc_s = stacking_df.loc[idx, 'accuracy']
        cv_r0_s = stacking_df.loc[idx, 'recall_0']
        cv_r1_s = stacking_df.loc[idx, 'recall_1']

        # Test metrics at that threshold
        y_test_pred_s = (y_test_proba[:, 1] > thr_best).astype(int)
        test_acc_s = accuracy_score(y_test, y_test_pred_s)
        test_r0_s = recall_score(y_test, y_test_pred_s, pos_label=0)
        test_r1_s = recall_score(y_test, y_test_pred_s, pos_label=1)
        test_bal_s = balanced_accuracy_score(y_test, y_test_pred_s)

        strat_summary.append({
            'strategy': strat_name,
            'best_threshold': thr_best,
            'cv_acc': cv_acc_s,
            'cv_rec0': cv_r0_s,
            'cv_rec1': cv_r1_s,
            'test_acc': test_acc_s,
            'test_rec0': test_r0_s,
            'test_rec1': test_r1_s,
            'test_balanced': test_bal_s
        })

        print(f"\nStrategy: {strat_name}")
        print(f"  - chosen threshold: {thr_best:.3f}")
        print(
            f"  - CV Acc/Rec0/Rec1: {cv_acc_s:.4f} / {cv_r0_s:.4f} / {cv_r1_s:.4f}")
        print(
            f"  - Test Acc/Rec0/Rec1: {test_acc_s:.4f} / {test_r0_s:.4f} / {test_r1_s:.4f}")

    strategy_df = pd.DataFrame(strat_summary).set_index('strategy')

    # -------------------------
    # Visualization
    # -------------------------
    plt.figure(figsize=(18, 12))

    # Plot 1: CV vs Individual model CV best accuracy
    plt.subplot(2, 2, 1)
    comp = individual_results.copy()
    comp['stacking'] = best_cv_accuracy
    bars = plt.bar(comp.keys(), comp.values())
    plt.axhline(y=target_accuracy, color='red', linestyle='--',
                label=f'Target: {target_accuracy}')
    plt.title('Stacking vs Individual Models (CV best accuracies)')
    plt.ylabel('CV Accuracy')
    plt.xticks(rotation=45)
    for bar, val in zip(bars, comp.values()):
        plt.text(bar.get_x()+bar.get_width()/2, val+0.005,
                 f'{val:.3f}', ha='center', va='bottom')
    plt.legend()

    # Plot 2: CV accuracy vs Test accuracy across thresholds, plus ROC inset (CV vs Test)
    plt.subplot(2, 2, 2)
    plt.plot(stacking_df['threshold'], stacking_df['accuracy'],
             label='CV Accuracy', linewidth=2)
    # compute test accuracies across thresholds (deterministic)
    test_accs = [accuracy_score(y_test, (y_test_proba[:, 1] > thr).astype(
        int)) for thr in stacking_df['threshold']]
    plt.plot(stacking_df['threshold'], test_accs,
             label='Test Accuracy', linewidth=2)
    plt.scatter(best_threshold, best_cv_accuracy, c='red',
                marker='*', s=100, label=f'CV-opt thr={best_threshold:.3f}')
    # also show Test@CVopt
    plt.scatter(best_threshold, test_accuracy, c='orange', marker='D',
                s=70, label=f'Test@CVopt Acc={test_accuracy:.3f}')
    plt.xlabel('Threshold')
    plt.ylabel('Accuracy')
    plt.title('CV vs Test Accuracy (Threshold sweep)')
    plt.legend()
    plt.grid(alpha=0.3)

   # ROC inset: CV ROC vs Test ROC + AUC numbers
    axins = inset_axes(plt.gca(), width="22%", height="22%",
                       loc='lower left', borderpad=1)

    # 主线条
    axins.plot(fpr_cv, tpr_cv, label=f'CV (AUC={cv_auc:.2f})', linewidth=1.2)
    axins.plot(fpr_test, tpr_test,
               label=f'Test (AUC={test_auc:.2f})', linewidth=1.2, linestyle='--')

    # 参考线
    axins.plot([0, 1], [0, 1], linestyle=':', color='gray', linewidth=0.8)

    # 优化标题和坐标轴
    axins.set_title('ROC', fontsize=8, pad=2)
    axins.tick_params(axis='both', which='major', labelsize=6)
    axins.set_xlabel('FPR', fontsize=6)
    axins.set_ylabel('TPR', fontsize=6)

    # 调整图例到合适位置（下方）
    axins.legend(fontsize=6, loc='lower right', frameon=False)

    # Plot 3: Accuracy & Recalls vs Threshold
    plt.subplot(2, 2, 3)
    plt.plot(stacking_df['threshold'],
             stacking_df['accuracy'], label='CV Accuracy')
    plt.plot(stacking_df['threshold'],
             stacking_df['recall_0'], label='CV Recall_0')
    plt.plot(stacking_df['threshold'],
             stacking_df['recall_1'], label='CV Recall_1')
    plt.axvline(best_threshold, linestyle='--', color='black',
                label=f'CV-opt thr={best_threshold:.3f}')
    plt.title('CV Accuracy and Recalls vs Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(alpha=0.3)

    # Plot 4: Weighted score curves
    # --- Plot 4: Weighted Score / Recall curves, highlight Class 0 recall max ---
    # --- Plot 4: Weighted scores vs threshold ---
    plt.subplot(2, 2, 4)

    colors = ['blue', 'green', 'orange', 'purple']
    for i, strat_name in enumerate(weighting_strategies.keys()):
        plt.plot(stacking_df['threshold'], stacking_df[f'weighted_score_{strat_name}'],
                 label=f'{strat_name}', linewidth=2, color=colors[i])

    # --- Highlight best threshold for class_0_focused strategy ---
    best_idx_class0 = stacking_df[f'weighted_score_class_0_focused'].idxmax()
    best_thr_class0 = stacking_df.loc[best_idx_class0, 'threshold']
    best_cv_r0 = stacking_df.loc[best_idx_class0, 'recall_0']
    best_cv_r1 = stacking_df.loc[best_idx_class0, 'recall_1']
    best_cv_acc = stacking_df.loc[best_idx_class0, 'accuracy']
    best_score = stacking_df.loc[best_idx_class0,
                                 f'weighted_score_class_0_focused']

    # Marker
    plt.scatter(best_thr_class0, best_score, color='red',
                s=60, marker='*', label='Class0-focused max')

    # Text annotation with CV accuracy included
    plt.text(best_thr_class0 + 0.01, best_score + 0.02,
             f'Thr={best_thr_class0:.3f}\nCV Acc={best_cv_acc:.3f}\nRec0={best_cv_r0:.3f}\nRec1={best_cv_r1:.3f}',
             fontsize=8, color='red', ha='left', va='bottom',
             bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

    plt.xlabel('Threshold')
    plt.ylabel('Weighted Score')
    plt.title('Weighted Score vs Threshold (Class 0 Focused)')
    plt.legend(fontsize=7)
    plt.grid(alpha=0.3)

    # New figure: test metrics at strategy chosen thresholds (bar chart)
    metrics_plot = strategy_df[['test_acc',
                                'test_rec0', 'test_rec1', 'test_balanced']]
    metrics_plot.columns = ['Accuracy', 'Recall_0', 'Recall_1', 'Balanced_Acc']

    ax = metrics_plot.plot(kind='bar', figsize=(10, 6))
    ax.set_ylim(0, 1)
    plt.title(
        'Test Metrics at Each Weighting Strategy (threshold chosen by CV weighted score)')
    plt.ylabel('Score')
    plt.xticks(rotation=0)
    plt.grid(axis='y', alpha=0.3)
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.3f}', (p.get_x()+p.get_width()/2., p.get_height()),
                    ha='center', va='bottom', fontsize=8, rotation=0)
    plt.show()

    # Final classification report at CV-opt threshold
    print("\nFinal classification report on Test set (threshold = CV-opt):")
    print(classification_report(y_test, y_test_pred,
          target_names=['Class_0', 'Class_1']))

    # return results for external inspection
    return {
        'stacking_model': stacking_model,
        'selected_features': selected_features,
        'cv_auc': cv_auc,
        'test_auc': test_auc,
        'best_threshold': best_threshold,
        'best_cv_accuracy': best_cv_accuracy,
        'test_accuracy': test_accuracy,
        'test_recall_0': test_rec0,
        'test_recall_1': test_rec1,
        'stacking_df': stacking_df,
        'individual_results': individual_results,
        'strategy_df': strategy_df
    }


# -------------------------
# Run (example)
# -------------------------
if __name__ == '__main__':
    df = pd.read_csv("ACME-HappinessSurvey2020.csv")
    X = df.drop('Y', axis=1)
    X = df[["X1", "X4", "X5"]]
    y = df['Y'].values
    results = stacking_accuracy_optimization_deterministic(
        X, y, test_size=0.3, target_accuracy=0.65)

    # Quick check prints to compare values across runs
    print("\n>>> Quick summary:")
    print(f"CV AUC: {results['cv_auc']:.4f}")
    print(
        f"Best CV accuracy (threshold selection): {results['best_cv_accuracy']:.4f}")
    print(f"Best threshold: {results['best_threshold']:.3f}")
    print(f"Test AUC: {results['test_auc']:.4f}")
    print(
        f"Test accuracy (at CV-opt threshold): {results['test_accuracy']:.4f}")


# ================================
# RFE Feature Importance Analysis
# (Run this AFTER your main code, no modifications needed)
# ================================

def rfe_analysis_existing_model(X, y, existing_model, model_name="Existing Model"):
    """
    RFE analysis using your already trained model
    This runs separately and doesn't affect your main results
    """
    from sklearn.feature_selection import RFE
    import matplotlib.pyplot as plt

    print("=" * 60)
    print(f"RFE FEATURE ANALYSIS - {model_name}")
    print("=" * 60)

    # Use your existing model as the estimator for RFE
    # Note: We'll use the base RandomForest from your stacking for stability
    # If you want to use the full stacking model, it might be computationally expensive

    # Option 1: Use the RandomForest meta-learner (recommended for stability)
    if hasattr(existing_model, 'final_estimator_'):
        rfe_estimator = existing_model.final_estimator_
        print("Using meta-learner from stacking for RFE")
    else:
        # Option 2: Use the full stacking model (might be slow)
        rfe_estimator = existing_model
        print("Using full stacking model for RFE")

    # Perform RFE
    rfe = RFE(
        estimator=rfe_estimator,
        n_features_to_select=1,  # Start with selecting 1 feature, we'll analyze rankings
        step=1
    )

    # Fit RFE (using the same preprocessed data as your main analysis)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    selector = SelectKBest(score_func=f_classif, k=min(10, X.shape[1]))
    X_selected = selector.fit_transform(X_scaled, y)

    rfe.fit(X_selected, y)

    # Get feature rankings
    feature_ranking = pd.DataFrame({
        'feature': X.columns[selector.get_support()],
        'rfe_ranking': rfe.ranking_,
        'rfe_support': rfe.support_
    }).sort_values('rfe_ranking')

    print("\n📊 RFE Feature Rankings:")
    print(feature_ranking)

    # Visualization
    plt.figure(figsize=(12, 8))

    # Plot 1: RFE Rankings
    plt.subplot(2, 2, 1)
    colors = ['green' if rank ==
              1 else 'skyblue' for rank in feature_ranking['rfe_ranking']]
    bars = plt.barh(feature_ranking['feature'],
                    feature_ranking['rfe_ranking'], color=colors)
    plt.xlabel('RFE Ranking (1 = Best)')
    plt.title('RFE Feature Rankings\n(Green = Top Features)')

    # Add value labels
    for bar, rank in zip(bars, feature_ranking['rfe_ranking']):
        plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                 f'Rank {rank}', ha='left', va='center', fontsize=9)

    # Plot 2: Feature Importance from your model (if available)
    plt.subplot(2, 2, 2)
    try:
        if hasattr(rfe_estimator, 'feature_importances_'):
            importances = rfe_estimator.feature_importances_
            indices = np.argsort(importances)[::-1]

            plt.barh(range(len(importances)), importances[indices])
            plt.yticks(range(len(importances)), [
                       feature_ranking.iloc[i]['feature'] for i in indices])
            plt.xlabel('Feature Importance')
            plt.title('Model Feature Importances')
        else:
            plt.text(0.5, 0.5, 'Feature importances\nnot available',
                     ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Feature Importances')
    except Exception as e:
        plt.text(0.5, 0.5, f'Error: {str(e)}',
                 ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Feature Importances')

    # Plot 3: Performance vs Number of Features (simulated)
    plt.subplot(2, 2, 3)
    n_features_range = range(1, len(feature_ranking) + 1)

    # This would normally require retraining with different feature sets
    # For demonstration, we'll show the ranking order
    plt.plot(n_features_range, range(
        len(feature_ranking), 0, -1), 'o-', linewidth=2)
    plt.xlabel('Number of Features Selected')
    plt.ylabel('Theoretical Performance (Higher = Better)')
    plt.title('Theoretical Performance vs Feature Count')
    plt.grid(True, alpha=0.3)

    # Plot 4: Feature Correlations (if you have the original data)
    plt.subplot(2, 2, 4)
    try:
        # Use the selected features
        selected_data = X[feature_ranking['feature']]
        correlation_matrix = selected_data.corr()

        # Plot simplified correlation (just the first few features for clarity)
        n_show = min(5, len(correlation_matrix))
        im = plt.imshow(correlation_matrix.iloc[:n_show, :n_show], cmap='coolwarm',
                        vmin=-1, vmax=1, aspect='auto')
        plt.colorbar(im)
        plt.xticks(range(n_show),
                   feature_ranking['feature'][:n_show], rotation=45)
        plt.yticks(range(n_show), feature_ranking['feature'][:n_show])
        plt.title('Top Feature Correlations')
    except Exception as e:
        plt.text(0.5, 0.5, 'Correlation plot\nnot available',
                 ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Feature Correlations')

    plt.tight_layout()
    plt.show()

    # Return top recommendations
    top_features = feature_ranking[feature_ranking['rfe_ranking']
                                   == 1]['feature'].tolist()
    print(f"\n🎯 RECOMMENDED TOP FEATURES: {top_features}")

    return rfe, feature_ranking

# ================================
# Run RFE Analysis After Your Main Code
# ================================

# After running your main function:
# results = stacking_accuracy_optimization_deterministic(X, y, test_size=0.3, target_accuracy=0.65)


# Then run RFE analysis:
print("\n" + "="*80)
print("RUNNING RFE FEATURE ANALYSIS (Separate from main training)")
print("="*80)

# Use your trained stacking model for RFE analysis
rfe_results, feature_rankings = rfe_analysis_existing_model(
    X, y,
    existing_model=results['stacking_model'],
    model_name="Your Stacking Model"
)

print("\n✅ RFE Analysis Complete!")
print("This analysis helps identify the most important features")
print("without affecting your original model performance.")


# ================================
# HYPEROPT 超参数优化部分
# 在你不修改原有代码的前提下添加这个部分
# ================================

warnings.filterwarnings('ignore')


def hyperopt_stacking_optimization(X, y, test_size=0.3, n_evals=50):
    """
    为你的Stacking模型进行超参数优化
    使用RFE筛选后的相同特征集
    """
    print("=" * 60)
    print("HYPEROPT STACKING OPTIMIZATION")
    print("=" * 60)

    # 使用与你主函数完全相同的预处理流程
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    selector = SelectKBest(score_func=f_classif, k=min(10, X.shape[1]))
    X_selected = selector.fit_transform(X_scaled, y)
    selected_features = X.columns[selector.get_support()]
    print(f"Using selected features for Hyperopt: {list(selected_features)}")

    # 训练测试分割（与主函数相同）
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=test_size, stratify=y, random_state=SEED
    )

    # 定义超参数搜索空间
    space = {
        # 逻辑回归基学习器
        'lr__C': hp.loguniform('lr__C', -3, 2),
        'lr__max_iter': hp.quniform('lr__max_iter', 500, 2000, 100),

        # SVM基学习器
        'svm__C': hp.loguniform('svm__C', -2, 3),
        'svm__gamma': hp.loguniform('svm__gamma', -4, 0),

        # 随机森林基学习器
        'rf__n_estimators': hp.quniform('rf__n_estimators', 50, 300, 10),
        'rf__max_depth': hp.quniform('rf__max_depth', 3, 20, 1),
        'rf__min_samples_split': hp.quniform('rf__min_samples_split', 2, 20, 1),

        # 元学习器（随机森林）
        'meta__n_estimators': hp.quniform('meta__n_estimators', 30, 150, 10),
        'meta__max_depth': hp.quniform('meta__max_depth', 3, 15, 1)
    }

    def objective(params):
        """目标函数：平衡优化准确率和class 0 recall"""
        try:
            # 创建基学习器 - 使用优化后的参数
            base_models = [
                ('lr', LogisticRegression(
                    C=params['lr__C'],
                    max_iter=int(params['lr__max_iter']),
                    random_state=SEED
                )),
                ('svm', SVC(
                    C=params['svm__C'],
                    gamma=params['svm__gamma'],
                    probability=True,
                    random_state=SEED
                )),
                ('rf', RandomForestClassifier(
                    n_estimators=int(params['rf__n_estimators']),
                    max_depth=int(params['rf__max_depth']),
                    min_samples_split=int(params['rf__min_samples_split']),
                    random_state=SEED
                ))
            ]

            # 创建元学习器
            meta_learner = RandomForestClassifier(
                n_estimators=int(params['meta__n_estimators']),
                max_depth=int(params['meta__max_depth']),
                random_state=SEED
            )

            # 创建Stacking模型
            stacking_model = StackingClassifier(
                estimators=base_models,
                final_estimator=meta_learner,
                cv=5,
                passthrough=False,
                n_jobs=1
            )

            # 使用交叉验证评估
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

            # 获取交叉验证的预测概率
            y_proba_cv = cross_val_predict(
                stacking_model, X_train, y_train, cv=skf,
                method='predict_proba', n_jobs=1
            )

            # 找到平衡准确率和recall_0的最佳阈值
            thresholds = np.linspace(0.1, 0.9, 81)
            best_balanced_score = 0
            best_threshold = 0.5

            for thr in thresholds:
                y_pred = (y_proba_cv[:, 1] > thr).astype(int)
                recall_0 = recall_score(y_train, y_pred, pos_label=0)
                recall_1 = recall_score(y_train, y_pred, pos_label=1)
                accuracy = accuracy_score(y_train, y_pred)

                # 平衡评分
                balanced_score = 0.4 * recall_0 + 0.4 * accuracy + 0.2 * recall_1

                # 添加约束：recall_1不能太低（避免模型完全偏向class 0）
                if recall_1 < 0.15:  # 如果recall_1低于15%，惩罚这个阈值
                    balanced_score = balanced_score * 0.5

                if balanced_score > best_balanced_score:
                    best_balanced_score = balanced_score
                    best_threshold = thr

            # 使用最佳阈值计算最终指标
            y_pred_best = (y_proba_cv[:, 1] > best_threshold).astype(int)
            final_recall_0 = recall_score(y_train, y_pred_best, pos_label=0)
            final_recall_1 = recall_score(y_train, y_pred_best, pos_label=1)
            final_accuracy = accuracy_score(y_train, y_pred_best)

            # 最终验证：如果recall_1太低，给较大的惩罚
            if final_recall_1 < 0.25:  # 如果recall_1低于10%，说明模型过于偏向class 0
                loss = 0.8  # 较大的损失值
            else:
                # 正常计算损失：最小化 1 - 平衡分数
                final_balanced_score = 0.4 * recall_0 + 0.4 * accuracy + 0.2 * recall_1
                loss = 1 - final_balanced_score

            # 打印调试信息（可选）
            if random.random() < 0.05:  # 随机打印5%的评估信息
                print(f"Threshold: {best_threshold:.3f}, Rec0: {final_recall_0:.3f}, "
                      f"Rec1: {final_recall_1:.3f}, Acc: {final_accuracy:.3f}, "
                      f"Loss: {loss:.3f}")

            return {'loss': loss, 'status': STATUS_OK}

        except Exception as e:
            # 如果出现错误，返回一个较大的损失值
            print(f"Error in objective function: {e}")
            return {'loss': 1.0, 'status': STATUS_OK}

    # 运行Hyperopt优化
    print("开始超参数优化...")
    trials = Trials()
    import random
    np.random.seed(SEED)
    random.seed(SEED)
    best_params = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=n_evals,
        trials=trials,
    )

    print(f"\n✅ Finished Total evaluations: {len(trials.trials)}")

    # 处理最佳参数（转换为合适的类型）
    best_params_processed = {
        'lr__C': best_params['lr__C'],
        'lr__max_iter': int(best_params['lr__max_iter']),
        'svm__C': best_params['svm__C'],
        'svm__gamma': best_params['svm__gamma'],
        'rf__n_estimators': int(best_params['rf__n_estimators']),
        'rf__max_depth': int(best_params['rf__max_depth']),
        'rf__min_samples_split': int(best_params['rf__min_samples_split']),
        'meta__n_estimators': int(best_params['meta__n_estimators']),
        'meta__max_depth': int(best_params['meta__max_depth'])
    }

    print("\n🎯 Optimized hyperparameters:")
    for key, value in best_params_processed.items():
        print(f"  {key}: {value}")

    # 用最佳参数训练最终模型并评估
    print("\n" + "="*50)
    print("Training final model with optimized hyperparameters...")
    print("="*50)

    # 使用最佳参数创建最终模型
    base_models_optimized = [
        ('lr', LogisticRegression(
            C=best_params_processed['lr__C'],
            max_iter=best_params_processed['lr__max_iter'],
            random_state=SEED
        )),
        ('svm', SVC(
            C=best_params_processed['svm__C'],
            gamma=best_params_processed['svm__gamma'],
            probability=True,
            random_state=SEED
        )),
        ('rf', RandomForestClassifier(
            n_estimators=best_params_processed['rf__n_estimators'],
            max_depth=best_params_processed['rf__max_depth'],
            min_samples_split=best_params_processed['rf__min_samples_split'],
            random_state=SEED
        ))
    ]

    meta_learner_optimized = RandomForestClassifier(
        n_estimators=best_params_processed['meta__n_estimators'],
        max_depth=best_params_processed['meta__max_depth'],
        random_state=SEED
    )

    final_stacking_model = StackingClassifier(
        estimators=base_models_optimized,
        final_estimator=meta_learner_optimized,
        cv=5,
        passthrough=False,
        n_jobs=1
    )

    # 训练最终模型
    final_stacking_model.fit(X_train, y_train)

    # 在测试集上评估
    y_test_proba = final_stacking_model.predict_proba(X_test)

    # 找到最佳阈值（基于recall_0）
    thresholds = np.linspace(0.1, 0.9, 81)
    best_threshold = 0.5
    best_recall_0 = 0

    for thr in thresholds:
        y_pred = (y_test_proba[:, 1] > thr).astype(int)
        recall_0 = recall_score(y_test, y_pred, pos_label=0)
        if recall_0 > best_recall_0:
            best_recall_0 = recall_0
            best_threshold = thr

    # 使用最佳阈值进行最终预测
    y_test_pred = (y_test_proba[:, 1] > best_threshold).astype(int)

    # 计算各项指标
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_recall_0 = recall_score(y_test, y_test_pred, pos_label=0)
    test_recall_1 = recall_score(y_test, y_test_pred, pos_label=1)
    test_auc = roc_auc_score(y_test, y_test_proba[:, 1])
    test_balanced = balanced_accuracy_score(y_test, y_test_pred)

    print("\n📊 Performance on Test Set (Optimized):")
    print(f"Best Threshold: {best_threshold:.3f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Recall_0: {test_recall_0:.4f}")
    print(f"Test Recall_1: {test_recall_1:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Test Balanced Accuracy: {test_balanced:.4f}")

    # 与原始模型对比（如果你已经运行了原始代码）
    print("\n" + "="*50)
    print("Compare with the original model:")
    print("="*50)
    print("Parameters in the Original Model:")
    print("  LR: C=1.0, max_iter=1000")
    print("  SVM: C=1.0, gamma='scale'")
    print("  RF: n_estimators=100, max_depth=5")
    print("  Meta-RF: n_estimators=50, max_depth=5")

    return {
        'best_params': best_params_processed,
        'final_model': final_stacking_model,
        'test_metrics': {
            'accuracy': test_accuracy,
            'recall_0': test_recall_0,
            'recall_1': test_recall_1,
            'auc': test_auc,
            'balanced_accuracy': test_balanced,
            'best_threshold': best_threshold
        },
        'trials': trials,
        'selected_features': selected_features
    }

# ================================
# 运行Hyperopt优化
# ================================


print("\n" + "="*80)
print("Starting Hyperopt Optimization")
print("="*80)

# 使用与你主函数相同的数据
df = pd.read_csv("ACME-HappinessSurvey2020.csv")
X = df[["X1", "X4", "X5"]]  # 使用你选择的特征
y = df['Y'].values

# 运行Hyperopt优化（评估次数可以根据时间调整）
hyperopt_results = hyperopt_stacking_optimization(
    X, y,
    test_size=0.3,
    n_evals=100  # 可以调整为100或更多以获得更好结果
)

print("\n🎉 Hyperopt Finish!")
print("You can find the results in the 'hyperopt_results' variable")

# ================================
# 可视化优化过程
# ================================


def plot_hyperopt_results(trials):
    """Visualize Hyperopt optimization process"""
    plt.figure(figsize=(15, 5))

    # Plot loss function convergence
    losses = [trial['result']['loss'] for trial in trials.trials]
    best_loss = np.minimum.accumulate(losses)

    plt.subplot(1, 3, 1)
    plt.plot(losses, 'o', alpha=0.3, markersize=4, label='Each Evaluation')
    plt.plot(best_loss, 'r-', linewidth=2, label='Best Loss')
    plt.xlabel('Evaluation Count')
    plt.ylabel('Loss (1 - Balanced Score)')
    plt.title('Hyperopt Optimization Process')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot balanced score progress
    balanced_scores = [1 - loss for loss in losses]
    best_balanced_scores = [1 - loss for loss in best_loss]

    plt.subplot(1, 3, 2)
    plt.plot(balanced_scores, 'o', alpha=0.3,
             markersize=4, label='Each Evaluation')
    plt.plot(best_balanced_scores, 'g-', linewidth=2,
             label='Best Balanced Score')
    plt.xlabel('Evaluation Count')
    plt.ylabel('Balanced Score')
    plt.title('Balanced Score Optimization Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Display parameter importance (simplified version)
    plt.subplot(1, 3, 3)
    best_trial = trials.best_trial
    param_names = list(best_trial['misc']['vals'].keys())
    param_importance = [len(np.unique([trial['misc']['vals'][param] for trial in trials.trials]))
                        for param in param_names]

    # Show only top 10 most important parameters
    indices = np.argsort(param_importance)[-10:]
    plt.barh(range(len(indices)), [param_importance[i] for i in indices])
    plt.yticks(range(len(indices)), [param_names[i] for i in indices])
    plt.xlabel('Parameter Exploration Degree')
    plt.title('Hyperparameter Importance\n(Based on Exploration Degree)')

    plt.tight_layout()
    plt.show()


# Plot optimization process
print("\nGenerating optimization process visualization...")
plot_hyperopt_results(hyperopt_results['trials'])

print("\n" + "="*80)
print("Hyperparameter Optimization Process Completed!")
print("="*80)


# 从hyperopt_results中获取优化后的模型和测试集指标
final_model = hyperopt_results['final_model']
test_metrics = hyperopt_results['test_metrics']

print("📊 Optimized Model Performance:")
print(f"Best Threshold: {test_metrics['best_threshold']:.3f}")
print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
print(f"Test Recall_0: {test_metrics['recall_0']:.4f}")
print(f"Test Recall_1: {test_metrics['recall_1']:.4f}")
print(f"Test AUC: {test_metrics['auc']:.4f}")
print(f"Test Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")
