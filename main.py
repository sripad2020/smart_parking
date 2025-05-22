import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier
import numpy as np
from xgboost import XGBRFClassifier,XGBClassifier
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_curve, auc, precision_recall_curve,
                             average_precision_score)
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder,label_binarize
from lazypredict.Supervised import LazyClassifier

data=pd.read_csv('IIoT_Smart_Parking_Management.csv')
print(data.columns)
print(data.info())
print(data.isna().sum())
print(data.describe())
print(len(data.columns))

print('------------------------------------')
print(data.Parking_Lot_Section.value_counts())
print('--------------------------------------')
print(data.Reserved_Status.value_counts())

data['Timestamp'] = pd.to_datetime(data['Timestamp']).dt.hour

for i in data.select_dtypes(include='object').columns.values:
    print(data[i].value_counts())

lab=LabelEncoder()
for i in data.select_dtypes(include='object').columns.values:
    data[i]=lab.fit_transform(data[i])

outlier=[]
for i in data.select_dtypes(include='number').columns.values:
    data['scores']=(data[i]-data[i].mean())/data[i].std()
    outliers=np.abs(data['scores']>3).sum()
    if outliers>0:
        outlier.append(i)

print(len(data))
thresh=3
for i in outlier:
    upper=data[i].mean()+thresh*data[i].std()
    lower=data[i].mean()-thresh*data[i].std()
    data=data[(data[i]>lower)&(data[i]<upper)]
print(len(data))




# Loop through each column in the dataset
for target_col in data.columns:
    # Drop self-correlation and find positively correlated features
    correlations = data.corr()[target_col].drop(target_col)
    correlated_features = [col for col in correlations.index if correlations[col] > 0]

    if not correlated_features:
        print(f"Skipping '{target_col}' — no positively correlated features.")
        continue

    X = data[correlated_features]
    y = data[target_col]
    print(f'-------------------{target_col}------------------')
    print('------------',X.columns,'--------------------')
    '''# Split the dataset
    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=42)

    print('----------------------------------', target_col, '--------------------------')

    # Define models
    models = {
        "Extra Trees": ExtraTreesClassifier(class_weight='balanced', random_state=42),
        "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        "XGBoost RF": XGBRFClassifier(n_estimators=200, max_depth=7, random_state=42)
    }

    # Fit and score each model
    for name, model in models.items():
        try:
            model.fit(x_train, y_train)
            score = model.score(x_test, y_test)
            print(f"{name}: {score:.4f}")
        except Exception as e:
            print(f"{name} failed: {e}")
'''
'''mod=LazyClassifier()
predictinos,modes=mod.fit(x_train,x_test,y_train,y_test)
print(predictinos)


models = {
    "Extra Trees": ExtraTreesClassifier(class_weight='balanced'),
    "Random Forest": RandomForestClassifier(class_weight='balanced'),
    "XGBoost": XGBClassifier(),
    "XGBoost RF": XGBRFClassifier(n_estimators=200, max_depth=7)
}

# 2. Train models and store results
results = {}
for name, model in models.items():
    model.fit(x_train, y_train)
    print(f"{name} ",model.score(x_test,y_test))
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(x_test)

    results[name] = {
        'model': model,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'score': model.score(x_test, y_test),
        'cm': confusion_matrix(y_test, y_pred),
        'cr': classification_report(y_test, y_pred, output_dict=True),
        'fpr': None,
        'tpr': None,
        'roc_auc': None,
        'precision': None,
        'recall': None,
        'average_precision': None
    }

    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    results[name]['fpr'] = fpr
    results[name]['tpr'] = tpr
    results[name]['roc_auc'] = roc_auc

    # Calculate precision-recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    average_precision = average_precision_score(y_test, y_proba)
    results[name]['precision'] = precision
    results[name]['recall'] = recall
    results[name]['average_precision'] = average_precision

    # Save models
    joblib.dump(model, f"{name.lower()+ "physico_chemical"}.pkl")

# =============================================
# VISUALIZATIONS
# =============================================

# 1. ROC Curve Comparison
fig_roc = go.Figure()
for name, res in results.items():
    fig_roc.add_trace(go.Scatter(
        x=res['fpr'], y=res['tpr'],
        mode='lines',
        name=f'{name} (AUC = {res["roc_auc"]:.2f})'
    ))
fig_roc.add_trace(go.Scatter(
    x=[0, 1], y=[0, 1],
    mode='lines',
    line=dict(dash='dash'),
    name='Random Chance'
))
fig_roc.update_layout(
    title='ROC Curve Comparison',
    xaxis_title='False Positive Rate',
    yaxis_title='True Positive Rate',
    yaxis=dict(scaleanchor="x", scaleratio=1),
    xaxis=dict(constrain='domain'),
    width=800, height=600,
    template='plotly_white'
)
fig_roc.show()

# 2. Precision-Recall Curve Comparison
fig_pr = go.Figure()
for name, res in results.items():
    fig_pr.add_trace(go.Scatter(
        x=res['recall'], y=res['precision'],
        mode='lines',
        name=f'{name} (AP = {res["average_precision"]:.2f})'
    ))
fig_pr.update_layout(
    title='Precision-Recall Curve Comparison',
    xaxis_title='Recall',
    yaxis_title='Precision',
    yaxis=dict(range=[0, 1.05]),
    xaxis=dict(range=[0, 1.05]),
    width=800, height=600,
    template='plotly_white'
)
fig_pr.show()

# 3. Confusion Matrix Heatmaps
fig_cm = make_subplots(
    rows=2, cols=2,
    subplot_titles=list(results.keys()),
    horizontal_spacing=0.15,
    vertical_spacing=0.15
)

for i, (name, res) in enumerate(results.items()):
    row = (i // 2) + 1
    col = (i % 2) + 1

    cm = res['cm']
    labels = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    values = cm.flatten()
    text = [f"{v}<br>{l}" for v, l in zip(values, labels)]

    fig_cm.add_trace(
        go.Heatmap(
            z=cm,
            x=['Predicted 0', 'Predicted 1'],
            y=['Actual 0', 'Actual 1'],
            text=text,
            texttemplate="%{text}",
            colorscale='Blues',
            showscale=False
        ),
        row=row, col=col
    )

fig_cm.update_layout(
    title_text='Confusion Matrices',
    height=700,
    width=900,
    template='plotly_white'
)
fig_cm.show()

# 4. Feature Importance Comparison (for tree-based models)
fig_feature_imp = make_subplots(
    rows=2, cols=2,
    subplot_titles=list(results.keys()),
    horizontal_spacing=0.15,
    vertical_spacing=0.2
)

for i, (name, res) in enumerate(results.items()):
    row = (i // 2) + 1
    col = (i % 2) + 1

    if hasattr(res['model'], 'feature_importances_'):
        importances = res['model'].feature_importances_
        indices = np.argsort(importances)[-10:]  # Top 10 features
        feature_names = [f'Feature {i}' for i in indices]  # Replace with actual feature names if available

        fig_feature_imp.add_trace(
            go.Bar(
                x=importances[indices],
                y=feature_names,
                orientation='h',
                name=name
            ),
            row=row, col=col
        )

fig_feature_imp.update_layout(
    title_text='Top 10 Feature Importances',
    height=700,
    width=900,
    showlegend=False,
    template='plotly_white'
)
fig_feature_imp.show()

# 5. Model Performance Metrics Comparison
metrics = ['precision', 'recall', 'f1-score']
classes = list(results[list(results.keys())[0]]['cr'].keys())[:-3]  # Exclude avg/total

fig_metrics = go.Figure()

for metric in metrics:
    metric_values = []
    for name in results:
        class_metrics = [results[name]['cr'][str(cls)][metric] for cls in classes]
        metric_values.append(np.mean(class_metrics))

    fig_metrics.add_trace(go.Bar(
        x=list(results.keys()),
        y=metric_values,
        name=metric.capitalize()
    ))

fig_metrics.update_layout(
    title='Average Classification Metrics by Model',
    xaxis_title='Model',
    yaxis_title='Score',
    barmode='group',
    height=600,
    width=800,
    template='plotly_white'
)
fig_metrics.show()

# 6. Calibration Curves
fig_calibration = go.Figure()

for name, res in results.items():
    prob_true, prob_pred = calibration_curve(y_test, res['y_proba'], n_bins=10)
    fig_calibration.add_trace(go.Scatter(
        x=prob_pred, y=prob_true,
        mode='lines+markers',
        name=name
    ))

fig_calibration.add_trace(go.Scatter(
    x=[0, 1], y=[0, 1],
    mode='lines',
    line=dict(dash='dash'),
    name='Perfectly Calibrated'
))

fig_calibration.update_layout(
    title='Calibration Curves',
    xaxis_title='Mean Predicted Probability',
    yaxis_title='Fraction of Positives',
    height=600,
    width=800,
    template='plotly_white'
)
fig_calibration.show()

# 7. Class Prediction Distribution
fig_pred_dist = go.Figure()

for name, res in results.items():
    fig_pred_dist.add_trace(go.Violin(
        x=[name] * len(res['y_proba']),
        y=res['y_proba'],
        name=name,
        box_visible=True,
        meanline_visible=True
    ))

fig_pred_dist.update_layout(
    title='Predicted Probability Distribution by Model',
    xaxis_title='Model',
    yaxis_title='Predicted Probability',
    height=600,
    width=800,
    template='plotly_white'
)
fig_pred_dist.show()

# 8. Accuracy Comparison
fig_accuracy = go.Figure(go.Bar(
    x=list(results.keys()),
    y=[res['score'] for res in results.values()],
    text=[f"{res['score']:.3f}" for res in results.values()],
    textposition='auto',
    marker_color=px.colors.qualitative.Plotly
))

fig_accuracy.update_layout(
    title='Model Accuracy Comparison',
    xaxis_title='Model',
    yaxis_title='Accuracy',
    height=600,
    width=800,
    template='plotly_white'
)
fig_accuracy.show()

# 9. Precision-Recall Tradeoff (Interactive)
fig_pr_tradeoff = go.Figure()

for name, res in results.items():
    fig_pr_tradeoff.add_trace(go.Scatter(
        x=res['recall'],
        y=res['precision'],
        mode='lines',
        name=name,
        customdata=np.stack((res['recall'], res['precision']), axis=-1),
        hovertemplate="Recall: %{customdata[0]:.2f}<br>Precision: %{customdata[1]:.2f}<extra></extra>"
    ))

fig_pr_tradeoff.update_layout(
    title='Precision-Recall Tradeoff',
    xaxis_title='Recall',
    yaxis_title='Precision',
    height=600,
    width=800,
    template='plotly_white'
)
fig_pr_tradeoff.show()

# 10. Model Decision Boundaries (2D PCA projection)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(x_test)

fig_decision = make_subplots(
    rows=2, cols=2,
    subplot_titles=list(results.keys()),
    horizontal_spacing=0.1,
    vertical_spacing=0.15
)

for i, (name, res) in enumerate(results.items()):
    row = (i // 2) + 1
    col = (i % 2) + 1

    # Create mesh grid
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                         np.linspace(y_min, y_max, 50))

    # Predict on mesh grid
    Z = res['model'].predict(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)

    fig_decision.add_trace(
        go.Contour(
            x=np.linspace(x_min, x_max, 50),
            y=np.linspace(y_min, y_max, 50),
            z=Z,
            colorscale=['red', 'blue'],
            opacity=0.3,
            showscale=False,
            name=name
        ),
        row=row, col=col
    )

    # Add actual data points
    for class_val, color in zip([0, 1], ['red', 'blue']):
        mask = y_test == class_val
        fig_decision.add_trace(
            go.Scatter(
                x=X_pca[mask, 0],
                y=X_pca[mask, 1],
                mode='markers',
                marker=dict(color=color),
                name=f'Class {class_val}',
                showlegend=(i == 0)
            ),
            row=row, col=col
        )

fig_decision.update_layout(
    title_text='Decision Boundaries (PCA Projection)',
    height=700,
    width=900,
    template='plotly_white'
)
fig_decision.show()

# 11. t-SNE Visualization with Model Predictions
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(x_test)

fig_tsne = make_subplots(
    rows=2, cols=2,
    subplot_titles=[f"{name} Predictions" for name in results.keys()],
    horizontal_spacing=0.1,
    vertical_spacing=0.15
)

for i, (name, res) in enumerate(results.items()):
    row = (i // 2) + 1
    col = (i % 2) + 1

    correct = (res['y_pred'] == y_test)
    incorrect = ~correct

    # Correct predictions
    fig_tsne.add_trace(
        go.Scatter(
            x=X_tsne[correct, 0],
            y=X_tsne[correct, 1],
            mode='markers',
            marker=dict(color='green', opacity=0.7),
            name='Correct',
            showlegend=(i == 0)
        ),
        row=row, col=col
    )

    # Incorrect predictions
    fig_tsne.add_trace(
        go.Scatter(
            x=X_tsne[incorrect, 0],
            y=X_tsne[incorrect, 1],
            mode='markers',
            marker=dict(color='red', opacity=0.7, symbol='x'),
            name='Incorrect',
            showlegend=(i == 0)
        ),
        row=row, col=col
    )

fig_tsne.update_layout(
    title_text='t-SNE Visualization with Model Predictions',
    height=700,
    width=900,
    template='plotly_white'
)
fig_tsne.show()


# Interactive Model Comparison Dashboard
metrics_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [res['score'] for res in results.values()],
    'Precision': [res['cr']['weighted avg']['precision'] for res in results.values()],
    'Recall': [res['cr']['weighted avg']['recall'] for res in results.values()],
    'F1-Score': [res['cr']['weighted avg']['f1-score'] for res in results.values()],
    'ROC AUC': [res['roc_auc'] for res in results.values()]
})

fig_dashboard = px.parallel_coordinates(
    metrics_df,
    color='Accuracy',
    dimensions=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC'],
    labels={'Accuracy': 'Accuracy', 'Precision': 'Precision',
            'Recall': 'Recall', 'F1-Score': 'F1-Score', 'ROC AUC': 'ROC AUC'},
    color_continuous_scale=px.colors.sequential.Viridis,
    title='Interactive Model Comparison Dashboard'
)

fig_dashboard.update_layout(
    height=600,
    width=900,
    template='plotly_white'
)
fig_dashboard.show()
'''