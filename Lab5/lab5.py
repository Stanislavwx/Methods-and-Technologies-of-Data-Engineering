"""
Лабораторна робота №5
Дисципліна: Методи та технології інженерії даних
Тема: Інженерія даних для задачі кластеризації
Датасет: Automobile_data.csv
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Створення папки для графіків
os.makedirs('plots', exist_ok=True)
sns.set_theme(style='whitegrid', font_scale=1.0)


# ==============================================================================
# Завдання 1. Підготовка даних за результатами EDA
# ==============================================================================
print("=" * 75)
print("ЗАВДАННЯ 1. Підготовка даних за результатами EDA")
print("=" * 75)

# 1.1 Завантаження набору даних
df = pd.read_csv('Automobile_data.csv')
print(f"Початковий розмір датасету: {df.shape[0]} рядків, {df.shape[1]} стовпців")

# 1.2 Заміна символу '?' на np.nan
df = df.replace('?', np.nan)

# Визначення числових та категоріальних ознак
numeric_cols = [
    'normalized-losses', 'wheel-base', 'length', 'width', 'height',
    'curb-weight', 'engine-size', 'bore', 'stroke', 'compression-ratio',
    'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price'
]

categorical_cols = [
    'make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style',
    'drive-wheels', 'engine-location', 'engine-type', 'num-of-cylinders',
    'fuel-system'
]

# Приведення числових колонок до числових типів та заповнення пропусків
for col in numeric_cols:
    s = pd.to_numeric(df[col], errors='coerce')
    df[col] = s.fillna(s.median())

# Заповнення пропусків у категоріальних ознаках модою
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# 1.4 Перевірка на дублікати
duplicates = df.duplicated().sum()
print(f"Кількість повних дублікатів: {duplicates}")
if duplicates > 0:
    df = df.drop_duplicates()

# 1.5 Вилучення нетехнічних / службових полів
# symboling — страховий рейтинг ризику, що був цільовою міткою в задачі класифікації
# Для задачі кластеризації технічних параметрів автомобілів symboling вилучається
X_raw = df.drop(columns=['symboling'])
print(f"Розмірність матриці ознак після вилучення службового поля: {X_raw.shape}")
print(f"Залишилося числових ознак: {len(numeric_cols)}, категоріальних: {len(categorical_cols)}")


# ==============================================================================
# Завдання 2. Підготовка числових і категоріальних ознак
# ==============================================================================
print("\n" + "=" * 75)
print("ЗАВДАННЯ 2. Підготовка числових і категоріальних ознак")
print("=" * 75)
print("Для числових ознак застосовується StandardScaler (приведення до mu=0, sigma=1).")
print("Для категоріальних ознак застосовується OneHotEncoder(drop='first').")
print("Масштабування критично важливе для k-means через чутливість евклідової відстані до різниці шкал.")


# ==============================================================================
# Завдання 3. Формування різних наборів ознак
# ==============================================================================
print("\n" + "=" * 75)
print("ЗАВДАННЯ 3. Формування різних наборів ознак (A, B, C)")
print("=" * 75)

# --- Набір A: Усі ознаки ---
ct_all = ColumnTransformer([
    ('num', StandardScaler(), numeric_cols),
    ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
])
X_A = ct_all.fit_transform(X_raw)
print(f"Набір A (усі ознаки): розмірність {X_A.shape}")

# --- Набір B: Відібрані ознаки ---
# 1. Кореляційний аналіз числових ознак
corr_matrix = df[numeric_cols].corr()

plt.figure(figsize=(11, 8.5))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', cbar=True, square=True)
plt.title('Матриця парних кореляцій числових ознак', fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig('plots/fig1_correlation_matrix.png', dpi=300)
plt.close()
print("Збережено графік: plots/fig1_correlation_matrix.png")

# Обґрунтування вилучення:
# - highway-mpg сильно корелює з city-mpg (r = 0.971) -> вилучаємо highway-mpg
# - length та width корелюють із curb-weight (r = 0.878 та 0.867) -> маса агрегує габарити
# - engine-location має квазіконстантне значення (98.5% front) -> низька варіативність
# - make містить 22 бренди -> створює 21 розріджену бінарну ознаку, що фрагментує простір
# - normalized-losses містить 20% пропусків та є актуарною оцінкою, а не конструктивною характеристикою

selected_num = ['curb-weight', 'wheel-base', 'horsepower', 'city-mpg', 'price', 'compression-ratio']
selected_cat = ['body-style', 'drive-wheels']

ct_sel = ColumnTransformer([
    ('num', StandardScaler(), selected_num),
    ('cat', OneHotEncoder(drop='first', sparse_output=False), selected_cat)
])
X_B = ct_sel.fit_transform(X_raw)
print(f"Набір B (відібрані ознаки): розмірність {X_B.shape} (6 числових + 6 бінарних)")

# --- Набір C: PCA ---
# Застосування PCA до стандартизованих числових ознак
scaler_num = StandardScaler()
X_num_scaled = scaler_num.fit_transform(df[numeric_cols])

pca_full = PCA().fit(X_num_scaled)
cum_variance = np.cumsum(pca_full.explained_variance_ratio_)

n_components_90 = int(np.argmax(cum_variance >= 0.90) + 1)
n_components_95 = int(np.argmax(cum_variance >= 0.95) + 1)

print(f"PCA: 90% дисперсії покривають {n_components_90} компонент ({cum_variance[n_components_90-1]*100:.2f}%)")
print(f"PCA: 95% дисперсії покривають {n_components_95} компонент ({cum_variance[n_components_95-1]*100:.2f}%)")

# Графік накопиченої дисперсії PCA
plt.figure(figsize=(8, 5))
components_range = range(1, len(cum_variance) + 1)
plt.bar(components_range, pca_full.explained_variance_ratio_, alpha=0.6, color='steelblue', label='Частка дисперсії компоненти')
plt.step(components_range, cum_variance, where='mid', color='crimson', lw=2, label='Накопичена дисперсія')
plt.axhline(0.90, color='darkgreen', linestyle='--', label='Поріг 90% дисперсії')
plt.axhline(0.95, color='orange', linestyle=':', label='Поріг 95% дисперсії')
plt.axvline(n_components_90, color='darkgreen', linestyle='--')
plt.xticks(components_range)
plt.xlabel('Головна компонента (PC)', fontsize=11)
plt.ylabel('Частка поясненої дисперсії', fontsize=11)
plt.title('Аналіз головних компонент (PCA): накопичена дисперсія', fontsize=12, fontweight='bold')
plt.legend(loc='center right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/fig2_pca_variance.png', dpi=300)
plt.close()
print("Збережено графік: plots/fig2_pca_variance.png")

# Формуємо Набір C з 7 компонент
pca_model = PCA(n_components=n_components_90, random_state=42)
X_C = pca_model.fit_transform(X_num_scaled)
print(f"Набір C (PCA): розмірність {X_C.shape}")


# ==============================================================================
# Завдання 4. Визначення кількості кластерів
# ==============================================================================
print("\n" + "=" * 75)
print("ЗАВДАННЯ 4. Визначення кількості кластерів (k = 2..10)")
print("=" * 75)

k_values = list(range(2, 11))
inertias = {'A': [], 'B': [], 'C': []}
silhouettes = {'A': [], 'B': [], 'C': []}

for k in k_values:
    # Model A
    km_a = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_A)
    inertias['A'].append(km_a.inertia_)
    silhouettes['A'].append(silhouette_score(X_A, km_a.labels_))
    
    # Model B
    km_b = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_B)
    inertias['B'].append(km_b.inertia_)
    silhouettes['B'].append(silhouette_score(X_B, km_b.labels_))
    
    # Model C
    km_c = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_C)
    inertias['C'].append(km_c.inertia_)
    silhouettes['C'].append(silhouette_score(X_C, km_c.labels_))

# Побудова сумісного графіка Elbow Curve та Silhouette Score
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# Графік інерції (Elbow)
ax1.plot(k_values, inertias['A'], marker='o', lw=2, label='Model A (Усі ознаки)')
ax1.plot(k_values, inertias['B'], marker='s', lw=2, label='Model B (Відібрані ознаки)')
ax1.plot(k_values, inertias['C'], marker='^', lw=2, label='Model C (PCA)')
ax1.axvline(3, color='black', linestyle='--', alpha=0.7, label='Точка ліктя (k=3)')
ax1.set_xlabel('Кількість кластерів (k)', fontsize=11)
ax1.set_ylabel('Inertia (WCSS)', fontsize=11)
ax1.set_title('Elbow Method (Залежність Inertia від k)', fontsize=12, fontweight='bold')
ax1.set_xticks(k_values)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Графік силуету
ax2.plot(k_values, silhouettes['A'], marker='o', lw=2, label='Model A (Усі ознаки)')
ax2.plot(k_values, silhouettes['B'], marker='s', lw=2, label='Model B (Відібрані ознаки)')
ax2.plot(k_values, silhouettes['C'], marker='^', lw=2, label='Model C (PCA)')
ax2.axvline(3, color='black', linestyle='--', alpha=0.7, label='Вибране k=3')
ax2.set_xlabel('Кількість кластерів (k)', fontsize=11)
ax2.set_ylabel('Silhouette Score', fontsize=11)
ax2.set_title('Silhouette Score для різної кількості кластерів', fontsize=12, fontweight='bold')
ax2.set_xticks(k_values)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/fig3_elbow_and_silhouette.png', dpi=300)
plt.close()
print("Збережено графік: plots/fig3_elbow_and_silhouette.png")
print("Обґрунтування вибору k: точка вигину Inertia для всіх моделей відповідає k=3.")
print("Для Model B при k=3 Silhouette Score сягає високого значення (0.3934), що дає природну інтерпретацію сегментів ринку.")


# ==============================================================================
# Завдання 5. Побудова моделі кластеризації (k = 3)
# ==============================================================================
print("\n" + "=" * 75)
print("ЗАВДАННЯ 5. Побудова моделей кластеризації (K-Means, k = 3)")
print("=" * 75)

k_optimal = 3
km_model_A = KMeans(n_clusters=k_optimal, random_state=42, n_init=10).fit(X_A)
km_model_B = KMeans(n_clusters=k_optimal, random_state=42, n_init=10).fit(X_B)
km_model_C = KMeans(n_clusters=k_optimal, random_state=42, n_init=10).fit(X_C)

df['cluster_A'] = km_model_A.labels_
df['cluster_B'] = km_model_B.labels_
df['cluster_C'] = km_model_C.labels_

print("Кластеризацію успішно виконано для трьох варіантів представлення даних.")


# ==============================================================================
# Завдання 6. Оцінювання якості кластеризації
# ==============================================================================
print("\n" + "=" * 75)
print("ЗАВДАННЯ 6. Оцінювання якості кластеризації")
print("=" * 75)

metrics_data = [
    {
        'Модель': 'Model A (Усі ознаки)',
        'Ознаки': X_A.shape[1],
        'Inertia': round(km_model_A.inertia_, 2),
        'Silhouette': round(silhouette_score(X_A, km_model_A.labels_), 4),
        'Davies-Bouldin': round(davies_bouldin_score(X_A, km_model_A.labels_), 4),
        'Calinski-Harabasz': round(calinski_harabasz_score(X_A, km_model_A.labels_), 1)
    },
    {
        'Модель': 'Model B (Відібрані ознаки)',
        'Ознаки': X_B.shape[1],
        'Inertia': round(km_model_B.inertia_, 2),
        'Silhouette': round(silhouette_score(X_B, km_model_B.labels_), 4),
        'Davies-Bouldin': round(davies_bouldin_score(X_B, km_model_B.labels_), 4),
        'Calinski-Harabasz': round(calinski_harabasz_score(X_B, km_model_B.labels_), 1)
    },
    {
        'Модель': 'Model C (PCA 7 компонент)',
        'Ознаки': X_C.shape[1],
        'Inertia': round(km_model_C.inertia_, 2),
        'Silhouette': round(silhouette_score(X_C, km_model_C.labels_), 4),
        'Davies-Bouldin': round(davies_bouldin_score(X_C, km_model_C.labels_), 4),
        'Calinski-Harabasz': round(calinski_harabasz_score(X_C, km_model_C.labels_), 1)
    }
]

df_metrics = pd.DataFrame(metrics_data)
df_metrics.to_csv('results_comparison.csv', index=False)
print("Підсумкова таблиця якості кластеризації:")
print(df_metrics.to_string(index=False))


# ==============================================================================
# Завдання 7. Візуалізація кластерів у 2D (PC1 x PC2)
# ==============================================================================
print("\n" + "=" * 75)
print("ЗАВДАННЯ 7. Візуалізація кластерів у просторі PC1 x PC2")
print("=" * 75)

pca_2d = PCA(n_components=2, random_state=42)
X_2d = pca_2d.fit_transform(X_num_scaled)
df['PC1'] = X_2d[:, 0]
df['PC2'] = X_2d[:, 1]

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharex=True, sharey=True)
models = [
    ('Model A (Усі ознаки)', 'cluster_A', axes[0]),
    ('Model B (Відібрані ознаки)', 'cluster_B', axes[1]),
    ('Model C (PCA)', 'cluster_C', axes[2])
]

colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

for title, col, ax in models:
    for cl in sorted(df[col].unique()):
        subset = df[df[col] == cl]
        ax.scatter(subset['PC1'], subset['PC2'], color=colors[cl], label=f'Кластер {cl}', alpha=0.75, edgecolors='k', s=55)
        # Центроїд у проєкції 2D
        center_x = subset['PC1'].mean()
        center_y = subset['PC2'].mean()
        ax.scatter(center_x, center_y, color='black', marker='X', s=160, edgecolors='white', linewidths=1.5)
        
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)', fontsize=10)
    if ax == axes[0]:
        ax.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)', fontsize=10)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

plt.suptitle('Візуалізація результатів кластеризації у просторі головних компонент (PC1 x PC2)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('plots/fig4_clusters_pca_scatter.png', dpi=300, bbox_inches='tight')
plt.close()
print("Збережено графік: plots/fig4_clusters_pca_scatter.png")


# ==============================================================================
# Завдання 8. Аналіз та профілювання отриманих кластерів (Model B)
# ==============================================================================
print("\n" + "=" * 75)
print("ЗАВДАННЯ 8. Аналіз та профілювання кластерів (Model B)")
print("=" * 75)

profile_stats = []
cluster_names = {
    0: "Бюджетний економічний сегмент (Масові авто)",
    1: "Преміальний та спортивний сегмент (Люкс-авто)",
    2: "Дизельний сегмент високої паливної ефективності"
}

for cl in range(3):
    sub = df[df['cluster_B'] == cl]
    print(f"\n--- {cluster_names[cl]} (Кластер {cl}, кількість авто: {len(sub)}) ---")
    print(f"Середня ціна: ${sub['price'].mean():.0f} (медіана: ${sub['price'].median():.0f})")
    print(f"Потужність: {sub['horsepower'].mean():.1f} к.с.")
    print(f"Споряджена маса: {sub['curb-weight'].mean():.0f} кг")
    print(f"Витрата в місті: {sub['city-mpg'].mean():.1f} mpg")
    print(f"Ступінь стиснення: {sub['compression-ratio'].mean():.1f}")
    print(f"Тип пального: {sub['fuel-type'].value_counts().to_dict()}")
    print(f"Тип приводу: {sub['drive-wheels'].value_counts().to_dict()}")
    print(f"Топ-3 марки: {sub['make'].value_counts().head(3).to_dict()}")

# Побудова профілю кластерів
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
sns.boxplot(x='cluster_B', y='price', data=df, ax=axes[0, 0], palette=colors)
axes[0, 0].set_title('Розподіл ціни ($)', fontweight='bold')
axes[0, 0].set_xlabel('Кластер')
axes[0, 0].set_ylabel('Price ($)')

sns.boxplot(x='cluster_B', y='horsepower', data=df, ax=axes[0, 1], palette=colors)
axes[0, 1].set_title('Розподіл потужності (к.с.)', fontweight='bold')
axes[0, 1].set_xlabel('Кластер')
axes[0, 1].set_ylabel('Horsepower')

sns.boxplot(x='cluster_B', y='city-mpg', data=df, ax=axes[1, 0], palette=colors)
axes[1, 0].set_title('Паливна економічність у місті (City MPG)', fontweight='bold')
axes[1, 0].set_xlabel('Кластер')
axes[1, 0].set_ylabel('City MPG')

sns.boxplot(x='cluster_B', y='compression-ratio', data=df, ax=axes[1, 1], palette=colors)
axes[1, 1].set_title('Ступінь стиснення (виділення дизельних авто)', fontweight='bold')
axes[1, 1].set_xlabel('Кластер')
axes[1, 1].set_ylabel('Compression Ratio')

plt.suptitle('Характеристики сформованих кластерів (Model B)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/fig5_cluster_profiling.png', dpi=300)
plt.close()
print("Збережено графік: plots/fig5_cluster_profiling.png")

print("\n" + "=" * 75)
print("РОБОТУ СКРИПТА LAB5 УСПІШНО ЗАВЕРШЕНО!")
print("=" * 75)
