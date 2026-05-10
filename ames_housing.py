import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from scipy import stats

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('AmesHousing.csv')
print(f"Исходный размер: {df.shape}")
print(f"Пропусков всего: {df.isnull().sum().sum()}")

cat_none = ['Alley', 'Pool QC', 'Fence', 'Misc Feature', 'Fireplace Qu',
            'Garage Type', 'Garage Finish', 'Garage Qual', 'Garage Cond',
            'Bsmt Qual', 'Bsmt Cond', 'Bsmt Exposure', 'BsmtFin Type 1',
            'BsmtFin Type 2', 'Mas Vnr Type', 'Electrical']
for col in cat_none:
    if col in df.columns:
        df[col] = df[col].fillna('None')

num_zero = ['Bsmt Full Bath', 'Bsmt Half Bath', 'BsmtFin SF 1', 'BsmtFin SF 2',
            'Bsmt Unf SF', 'Total Bsmt SF', 'Garage Area', 'Garage Cars',
            'Mas Vnr Area']
for col in num_zero:
    if col in df.columns:
        df[col] = df[col].fillna(0)

if 'Lot Frontage' in df.columns and 'Neighborhood' in df.columns:
    df['Lot Frontage'] = df.groupby('Neighborhood')['Lot Frontage'].transform(
        lambda x: x.fillna(x.median()))
    df['Lot Frontage'] = df['Lot Frontage'].fillna(df['Lot Frontage'].median())

for col in df.select_dtypes(include=[np.number]).columns:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

for col in df.select_dtypes(include=['object']).columns:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mode()[0])

print(f"После очистки пропусков: {df.isnull().sum().sum()}")

df['Age_at_Sale'] = df['Yr Sold'] - df['Year Built']
df['Years_Since_Remodel'] = df['Yr Sold'] - df['Year Remod/Add']
df['Years_Since_Remodel'] = df['Years_Since_Remodel'].clip(lower=0)

target = 'SalePrice'
y = df[target]
X = df.drop(columns=[target, 'Order', 'PID'], errors='ignore')

cat_cols = X.select_dtypes(include=['object']).columns.tolist()
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

time_cols = ['Mo Sold', 'Yr Sold']
num_cols_for_model = [c for c in num_cols if c not in time_cols]

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols_for_model),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
])

X_processed = preprocessor.fit_transform(X)

ridge = Ridge(alpha=1.0, random_state=42)
ridge.fit(X_processed, y)

cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols)
all_feature_names = num_cols_for_model + list(cat_feature_names)

coef_df = pd.DataFrame({
    'feature': all_feature_names,
    'coef': ridge.coef_
})
coef_df['abs_coef'] = coef_df['coef'].abs()
top10 = coef_df.nlargest(10, 'abs_coef')

plt.figure(figsize=(12, 6))
plt.barh(top10['feature'], top10['abs_coef'], color='steelblue')
plt.title('Топ-10 важных признаков')
plt.xlabel('Коэффициент')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('top10_features.png', dpi=100, bbox_inches='tight')
plt.close()

print("\nТоп-10 признаков:")
print(top10[['feature', 'abs_coef']].to_string(index=False))

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(df['Gr Liv Area'], y, alpha=0.5, s=20, c='steelblue')
plt.title('Цена vs Жилая площадь (до очистки)')
plt.xlabel('Gr Liv Area (кв. футы)')
plt.ylabel('SalePrice ($)')

df_temp = df[['Gr Liv Area', target]].copy()
df_temp['z_score'] = np.abs(stats.zscore(df_temp[target]))
outliers_z = df_temp[df_temp['z_score'] > 3]

Q1 = y.quantile(0.25)
Q3 = y.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers_iqr = df[(y < lower_bound) | (y > upper_bound)]

iso = IsolationForest(contamination=0.05, random_state=42)
X_price_area = df[['Gr Liv Area', target]].values
iso_labels = iso.fit_predict(X_price_area)
outliers_if = df[iso_labels == -1]

outlier_indices = set(outliers_z.index) | set(outliers_iqr.index) | set(outliers_if.index)
print(f"\nАномалий Z-score: {len(outliers_z)}")
print(f"Аномалий IQR: {len(outliers_iqr)}")
print(f"Аномалий Isolation Forest: {len(outliers_if)}")
print(f"Всего уникальных аномалий: {len(outlier_indices)}")

df_clean = df.drop(index=outlier_indices)
y_clean = df_clean[target]
X_clean = df_clean.drop(columns=[target, 'Order', 'PID'], errors='ignore')

plt.subplot(1, 2, 2)
plt.scatter(df_clean['Gr Liv Area'], y_clean, alpha=0.5, s=20, c='green')
plt.title('Цена vs Жилая площадь (после очистки)')
plt.xlabel('Gr Liv Area (кв. футы)')
plt.ylabel('SalePrice ($)')
plt.tight_layout()
plt.savefig('price_vs_area_outliers.png', dpi=100, bbox_inches='tight')
plt.close()

X_clean_processed = preprocessor.fit_transform(X_clean)
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_clean_processed, y_clean, test_size=0.2, random_state=42)

ridge_before = Ridge(alpha=1.0)
ridge_before.fit(X_train, y_train)
y_pred_before = ridge_before.predict(X_test)
r2_before = r2_score(y_test, y_pred_before)
rmse_before = np.sqrt(mean_squared_error(y_test, y_pred_before))

ridge_after = Ridge(alpha=1.0)
ridge_after.fit(X_train_c, y_train_c)
y_pred_after = ridge_after.predict(X_test_c)
r2_after = r2_score(y_test_c, y_pred_after)
rmse_after = np.sqrt(mean_squared_error(y_test_c, y_pred_after))

print(f"\nДо удаления аномалий: R²={r2_before:.4f}, RMSE=${rmse_before:,.0f}")
print(f"После удаления аномалий: R²={r2_after:.4f}, RMSE=${rmse_after:,.0f}")


cluster_features = df_clean[num_cols_for_model].copy()
cluster_scaled = StandardScaler().fit_transform(cluster_features)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
clusters = kmeans.fit_predict(cluster_scaled)
df_clean['Cluster'] = clusters

print(f"\nРазмеры кластеров:")
print(df_clean['Cluster'].value_counts().sort_index())

print("\nСредние характеристики кластеров:")
print(df_clean.groupby('Cluster')[['Gr Liv Area', 'Age_at_Sale', 'Lot Area', 'Overall Qual']].mean())

pca_viz = PCA(n_components=2)
cluster_2d = pca_viz.fit_transform(cluster_scaled)

plt.figure(figsize=(10, 8))
for c in range(4):
    mask = clusters == c
    plt.scatter(cluster_2d[mask, 0], cluster_2d[mask, 1], s=10, alpha=0.6, label=f'Кластер {c}')
plt.title('Кластеры недвижимости (PCA, 2D)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.tight_layout()
plt.savefig('clusters_pca.png', dpi=100, bbox_inches='tight')
plt.close()

pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_processed)
print(f"\nPCA: {X_processed.shape[1]} признаков -> {X_pca.shape[1]} компонент")

X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

ridge_pca = Ridge(alpha=1.0)
ridge_pca.fit(X_train_pca, y_train)
y_pred_pca = ridge_pca.predict(X_test_pca)
print(f"Ridge на PCA: R²={r2_score(y_test, y_pred_pca):.4f}, RMSE=${np.sqrt(mean_squared_error(y_test, y_pred_pca)):,.0f}")

df_clean['Sale_Date'] = pd.to_datetime(
    df_clean['Yr Sold'].astype(int).astype(str) + '-' +
    df_clean['Mo Sold'].astype(int).astype(str) + '-01'
)

monthly_prices = df_clean.groupby('Mo Sold')['SalePrice'].agg(['mean', 'std', 'count'])
monthly_prices = monthly_prices.reset_index()

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.errorbar(monthly_prices['Mo Sold'], monthly_prices['mean'],
             yerr=monthly_prices['std']/np.sqrt(monthly_prices['count']),
             marker='o', capsize=5, color='steelblue')
plt.title('Средняя цена по месяцам (сезонность)')
plt.xlabel('Месяц')
plt.ylabel('Средняя цена ($)')
plt.xticks(range(1, 13), ['Янв','Фев','Мар','Апр','Май','Июн','Июл','Авг','Сен','Окт','Ноя','Дек'])
plt.grid(True, alpha=0.3)

yearly_prices = df_clean.groupby('Yr Sold')['SalePrice'].agg(['mean', 'std', 'count'])
yearly_prices = yearly_prices.reset_index()

plt.subplot(1, 2, 2)
plt.errorbar(yearly_prices['Yr Sold'], yearly_prices['mean'],
             yerr=yearly_prices['std']/np.sqrt(yearly_prices['count']),
             marker='o', capsize=5, color='coral', linewidth=2)
plt.title('Средняя цена по годам (кризис 2008)')
plt.xlabel('Год')
plt.ylabel('Средняя цена ($)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('time_analysis.png', dpi=100, bbox_inches='tight')
plt.close()

print("\nСРЕДНИЕ ЦЕНЫ ПО ГОДАМ")
for _, row in yearly_prices.iterrows():
    print(f"  {int(row['Yr Sold'])}: ${row['mean']:,.0f} ± ${row['std']:,.0f} ({int(row['count'])} продаж)")

print("\nСРЕДНИЕ ЦЕНЫ ПО МЕСЯЦАМ")
spring = [3, 4, 5]
winter = [12, 1, 2]
spring_price = df_clean[df_clean['Mo Sold'].isin(spring)]['SalePrice'].mean()
winter_price = df_clean[df_clean['Mo Sold'].isin(winter)]['SalePrice'].mean()
print(f"  Весна (март-май): ${spring_price:,.0f}")
print(f"  Зима (дек-фев): ${winter_price:,.0f}")
print(f"  Разница: ${spring_price - winter_price:,.0f} ({(spring_price/winter_price - 1)*100:.1f}%)")

price_2007 = yearly_prices[yearly_prices['Yr Sold'] == 2007]['mean'].values[0]
price_2008 = yearly_prices[yearly_prices['Yr Sold'] == 2008]['mean'].values[0]
price_2009 = yearly_prices[yearly_prices['Yr Sold'] == 2009]['mean'].values[0]

print(f"\nКРИЗИС 2008")
print(f"  Цена 2007: ${price_2007:,.0f}")
print(f"  Цена 2008: ${price_2008:,.0f}")
print(f"  Цена 2009: ${price_2009:,.0f}")
if price_2008 < price_2007:
    print(f"  Падение 2007->2008: ${price_2007 - price_2008:,.0f} ({(1 - price_2008/price_2007)*100:.1f}%)")
if price_2009 < price_2008:
    print(f"  Падение 2008->2009: ${price_2008 - price_2009:,.0f} ({(1 - price_2009/price_2008)*100:.1f}%)")

print("Созданы: top10_features.png, price_vs_area_outliers.png, clusters_pca.png, time_analysis.png")