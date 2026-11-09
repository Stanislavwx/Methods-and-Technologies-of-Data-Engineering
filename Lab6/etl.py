import os
import sys
import json
import time
import datetime
import argparse
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import seaborn as sns


# ==============================================================================
# КОНФІГУРАЦІЯ ТА ШЛЯХИ
# ==============================================================================
RAW_DIR = 'raw'
PROCESSED_DIR = 'processed'
PLOTS_DIR = 'plots'

PRODUCTS_RAW_PATH = os.path.join(RAW_DIR, 'products.csv')
RATES_RAW_PATH = os.path.join(RAW_DIR, 'exchange_rates.json')
PRODUCTS_PROCESSED_CSV = os.path.join(PROCESSED_DIR, 'products_processed.csv')
PRODUCTS_PROCESSED_PARQUET = os.path.join(PROCESSED_DIR, 'products_processed.parquet')

# Створення необхідних директорій
for directory in [RAW_DIR, PROCESSED_DIR, PLOTS_DIR]:
    os.makedirs(directory, exist_ok=True)


# ==============================================================================
# 1. ОТРИМАННЯ ДАНИХ (EXTRACT)
# ==============================================================================
def extract_products_data() -> pd.DataFrame:
    """
    Скрапінг каталогу книг із відкритого сайту books.toscrape.com
    з використанням BeautifulSoup.
    """
    print("[EXTRACT] Отримання даних про товари з сайту books.toscrape.com...")
    base_url = 'http://books.toscrape.com/'
    
    categories_to_scrape = [
        ('Travel', 'catalogue/category/books/travel_2/index.html'),
        ('Mystery', 'catalogue/category/books/mystery_3/index.html'),
        ('Historical Fiction', 'catalogue/category/books/historical-fiction_4/index.html'),
        ('Sequential Art', 'catalogue/category/books/sequential-art_5/index.html'),
        ('Classics', 'catalogue/category/books/classics_6/index.html'),
        ('Philosophy', 'catalogue/category/books/philosophy_7/index.html'),
        ('Romance', 'catalogue/category/books/romance_8/index.html'),
        ('Fiction', 'catalogue/category/books/fiction_10/index.html'),
        ('Science Fiction', 'catalogue/category/books/science-fiction_16/index.html'),
        ('Business', 'catalogue/category/books/business_35/index.html')
    ]

    rating_map = {'One': 1, 'Two': 2, 'Three': 3, 'Four': 4, 'Five': 5}
    scraped_records = []
    scrape_timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    for category_name, rel_path in categories_to_scrape:
        current_url = base_url + rel_path
        while current_url:
            response = requests.get(current_url, timeout=10)
            response.encoding = 'utf-8'
            soup = BeautifulSoup(response.text, 'html.parser')
            
            pods = soup.select('article.product_pod')
            for pod in pods:
                title = pod.h3.a['title'].strip()
                price_str = pod.select_one('.price_color').text.strip()
                price_val = float(price_str.replace('£', '').replace('Â', '').strip())
                
                classes = pod.select_one('.star-rating')['class']
                rating_str = [c for c in classes if c != 'star-rating'][0]
                rating_val = rating_map.get(rating_str, 3)
                
                availability = pod.select_one('.availability').text.strip()

                scraped_records.append({
                    'product_name': title,
                    'category': category_name,
                    'price': price_val,
                    'currency': 'GBP',
                    'rating': rating_val,
                    'availability': availability,
                    'scraped_at': scrape_timestamp
                })

            next_page = soup.select_one('li.next a')
            if next_page:
                parent_dir = '/'.join(current_url.split('/')[:-1])
                current_url = parent_dir + '/' + next_page['href']
            else:
                current_url = None

    df_products = pd.DataFrame(scraped_records)
    df_products.to_csv(PRODUCTS_RAW_PATH, index=False)
    print(f"[EXTRACT] Збережено {len(df_products)} записів у {PRODUCTS_RAW_PATH}")
    return df_products


def extract_exchange_rates() -> dict:
    """
    Отримання офіційних курсів валют (USD, EUR, GBP) від API НБУ.
    """
    print("[EXTRACT] Запит курсів валют з API Національного банку України...")
    nbu_url = 'https://bank.gov.ua/NBUStatService/v1/statdirectory/exchange?json'
    response = requests.get(nbu_url, timeout=10)
    data = response.json()

    target_currencies = {'USD', 'EUR', 'GBP'}
    rates = {
        item['cc']: {
            'rate': float(item['rate']),
            'currency_name': item['txt'],
            'date': item['exchangedate']
        }
        for item in data if item['cc'] in target_currencies
    }

    with open(RATES_RAW_PATH, 'w', encoding='utf-8') as f:
        json.dump(rates, f, ensure_ascii=False, indent=2)

    print(f"[EXTRACT] Курси валют успішно збережено у {RATES_RAW_PATH}:")
    for cc, info in rates.items():
        print(f"  • {cc} ({info['currency_name']}): {info['rate']:.4f} UAH (на {info['date']})")
    return rates


def run_extract(force_fetch: bool = False):
    """Виконання етапу Extract (з кешуванням або примусовим запитом)"""
    print("\n" + "=" * 70)
    print("ЕТАП 1. Отримання даних (Extract)")
    print("=" * 70)

    if force_fetch or not os.path.exists(PRODUCTS_RAW_PATH):
        df_raw = extract_products_data()
    else:
        print(f"[EXTRACT] Використано збережений файл товарів: {PRODUCTS_RAW_PATH}")
        df_raw = pd.read_csv(PRODUCTS_RAW_PATH)

    if force_fetch or not os.path.exists(RATES_RAW_PATH):
        rates = extract_exchange_rates()
    else:
        print(f"[EXTRACT] Використано збережені курси валют: {RATES_RAW_PATH}")
        with open(RATES_RAW_PATH, 'r', encoding='utf-8') as f:
            rates = json.load(f)

    print(f"[EXTRACT] Успішно завантажено {len(df_raw)} рядків і {len(rates)} валют.")
    return df_raw, rates


# ==============================================================================
# 2. ПЕРЕТВОРЕННЯ ТА ІНТЕГРАЦІЯ ДАНИХ (TRANSFORM)
# ==============================================================================
def run_transform(df_raw: pd.DataFrame, rates: dict) -> pd.DataFrame:
    """
    Очищення, стандартизація типів, об'єднання з курсами валют та
    мультивалютний розрахунок (GBP -> UAH -> USD, EUR).
    """
    print("\n" + "=" * 70)
    print("ЕТАП 2. Перетворення та інтеграція даних (Transform)")
    print("=" * 70)

    df = df_raw.copy()

    # 1. Перевірка та видалення дублікатів
    initial_count = len(df)
    df = df.drop_duplicates()
    dups_removed = initial_count - len(df)
    print(f"[TRANSFORM] Перевірка дублікатів: вилучено {dups_removed} дублікатів (залишилось {len(df)})")

    # 2. Перевірка наявності пропусків
    missing = df.isnull().sum()
    print("[TRANSFORM] Перевірка пропущених значень у ключових полях:")
    for col, count in missing.items():
        print(f"  • {col}: {count} пропусків")

    # 3. Приведення числових та часових типів
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['scraped_at'] = pd.to_datetime(df['scraped_at'])

    # Очищення можливих артефактів кодування в текстових назвах
    def clean_text(text):
        if not isinstance(text, str):
            return text
        try:
            return text.encode('latin1').decode('utf-8')
        except Exception:
            return text
    df['product_name'] = df['product_name'].apply(clean_text)

    # 4. Інтеграція курсів валют
    rate_gbp = float(rates['GBP']['rate'])
    rate_usd = float(rates['USD']['rate'])
    rate_eur = float(rates['EUR']['rate'])

    # 5. Розрахунок цін у трьох валютах (UAH, USD, EUR)
    df['price_gbp'] = df['price'].round(2)
    df['price_uah'] = (df['price_gbp'] * rate_gbp).round(2)
    df['price_usd'] = (df['price_uah'] / rate_usd).round(2)
    df['price_eur'] = (df['price_uah'] / rate_eur).round(2)

    df['rate_gbp_uah'] = rate_gbp
    df['rate_usd_uah'] = rate_usd
    df['rate_eur_uah'] = rate_eur

    # Видалення застарілих сирих колонок
    cols_to_drop = [c for c in ['price', 'currency'] if c in df.columns]
    df_processed = df.drop(columns=cols_to_drop)

    # 6. Валідація коректності перерахунку валют
    check_usd = np.abs(df_processed['price_usd'] * rate_usd - df_processed['price_uah'])
    check_eur = np.abs(df_processed['price_eur'] * rate_eur - df_processed['price_uah'])
    max_err_usd = check_usd.max()
    max_err_eur = check_eur.max()

    print(f"[TRANSFORM] Контроль коректності конвертації:")
    print(f"  • Максимальна похибка USD -> UAH: {max_err_usd:.4f} ₴ (в межах заокруглення до копійок)")
    print(f"  • Максимальна похибка EUR -> UAH: {max_err_eur:.4f} ₴ (в межах заокруглення до копійок)")

    assert (check_usd < 1.0).all(), "Критична помилка перерахунку USD!"
    assert (check_eur < 1.0).all(), "Критична помилка перерахунку EUR!"
    print("[TRANSFORM] Усі перевірки валідності перерахунку успішно пройдено.")

    return df_processed


# ==============================================================================
# 3. ЗАВАНТАЖЕННЯ ТА ПОРІВНЯННЯ ФОРМАТІВ (LOAD)
# ==============================================================================
def run_load_and_benchmark(df_processed: pd.DataFrame):
    """
    Збереження підготовленого набору в CSV та Parquet,
    повторне завантаження та бенчмарк (розмір, типи, час читання).
    """
    print("\n" + "=" * 70)
    print("ЕТАП 3. Завантаження даних та бенчмарк форматів (Load)")
    print("=" * 70)

    # Збереження у двох форматах
    df_processed.to_csv(PRODUCTS_PROCESSED_CSV, index=False)
    df_processed.to_parquet(PRODUCTS_PROCESSED_PARQUET, engine='pyarrow', index=False)
    print(f"[LOAD] Збережено CSV: {PRODUCTS_PROCESSED_CSV}")
    print(f"[LOAD] Збережено Parquet: {PRODUCTS_PROCESSED_PARQUET}")

    # 1. Порівняння розмірів файлів
    size_csv = os.path.getsize(PRODUCTS_PROCESSED_CSV)
    size_parquet = os.path.getsize(PRODUCTS_PROCESSED_PARQUET)
    ratio_size = size_csv / size_parquet

    print(f"\n[LOAD] Порівняння розміру на диску:")
    print(f"  • CSV:     {size_csv:>6} байт ({size_csv / 1024:.2f} КБ)")
    print(f"  • Parquet: {size_parquet:>6} байт ({size_parquet / 1024:.2f} КБ)")
    print(f"  • Parquet компактніший у {ratio_size:.2f}x разів (-{(1 - 1/ratio_size)*100:.1f}%)")

    # 2. Повторне зчитування та порівняння часу
    n_iterations = 100
    t0 = time.perf_counter()
    for _ in range(n_iterations):
        _ = pd.read_csv(PRODUCTS_PROCESSED_CSV)
    time_csv_ms = (time.perf_counter() - t0) / n_iterations * 1000

    t0 = time.perf_counter()
    for _ in range(n_iterations):
        _ = pd.read_parquet(PRODUCTS_PROCESSED_PARQUET)
    time_parquet_ms = (time.perf_counter() - t0) / n_iterations * 1000

    print(f"\n[LOAD] Порівняння середнього часу читання ({n_iterations} ітерацій):")
    print(f"  • pd.read_csv():     {time_csv_ms:.3f} мс")
    print(f"  • pd.read_parquet(): {time_parquet_ms:.3f} мс")

    # 3. Порівняння збереження типів даних
    df_csv_loaded = pd.read_csv(PRODUCTS_PROCESSED_CSV)
    df_parquet_loaded = pd.read_parquet(PRODUCTS_PROCESSED_PARQUET)

    comparison_table = pd.DataFrame({
        'Original': df_processed.dtypes.astype(str),
        'Reloaded CSV': df_csv_loaded.dtypes.astype(str),
        'Reloaded Parquet': df_parquet_loaded.dtypes.astype(str)
    })
    print("\n[LOAD] Порівняння збереження типів даних (dtypes):")
    print(comparison_table)

    return {
        'csv_size_kb': size_csv / 1024,
        'parquet_size_kb': size_parquet / 1024,
        'compression_ratio': ratio_size,
        'time_csv_ms': time_csv_ms,
        'time_parquet_ms': time_parquet_ms,
        'dtypes_comparison': comparison_table
    }


# ==============================================================================
# 4. BI-АНАЛІТИКА
# ==============================================================================
def run_bi_analytics(df: pd.DataFrame):
    """
    Розрахунок аналітичних показників та генерація підсумкових таблиць.
    """
    print("\n" + "=" * 70)
    print("ЕТАП 4. BI-Аналітика та підсумкові таблиці")
    print("=" * 70)

    # 1. Середня та медіанна ціна за категоріями
    m1 = df.groupby('category')['price_uah'].agg(
        Товарів='count',
        Середня_ціна_UAH='mean',
        Медіанна_ціна_UAH='median',
        Станд_відхилення='std'
    ).reset_index()
    m1['Середня_ціна_UAH'] = m1['Середня_ціна_UAH'].round(2)
    m1['Медіанна_ціна_UAH'] = m1['Медіанна_ціна_UAH'].round(2)
    m1['Станд_відхилення'] = m1['Станд_відхилення'].round(2)
    print("\n[BI-1] Середня та медіанна вартість товарів за категоріями:")
    print(m1.to_string(index=False))

    # 2. Мінімальна, максимальна ціна та діапазон
    m2 = df.groupby('category')['price_uah'].agg(
        Мін_ціна_UAH='min',
        Макс_ціна_UAH='max'
    ).reset_index()
    m2['Діапазон_UAH'] = (m2['Макс_ціна_UAH'] - m2['Мін_ціна_UAH']).round(2)
    print("\n[BI-2] Екстремуми цін (Min / Max) та ціновий розмах за категоріями:")
    print(m2.to_string(index=False))

    # 3. Розподіл товарів за ціновими діапазонами
    bins = [0, 1500, 2500, float('inf')]
    labels = ['Бюджетний (< 1500 ₴)', 'Середній (1500–2500 ₴)', 'Преміум (> 2500 ₴)']
    df['ціновий_сегмент'] = pd.cut(df['price_uah'], bins=bins, labels=labels)
    m3 = df['ціновий_сегмент'].value_counts()[labels].reset_index()
    m3.columns = ['Ціновий_сегмент', 'Кількість_товарів']
    m3['Частка_%'] = (m3['Кількість_товарів'] / len(df) * 100).round(2)
    print("\n[BI-3] Структура каталогу за ціновими сегментами:")
    print(m3.to_string(index=False))

    # 4. Частка категорій у каталозі та сумарна вартість
    m4 = df.groupby('category').agg(
        Кількість=('product_name', 'count'),
        Сумарна_вартість_UAH=('price_uah', 'sum')
    ).reset_index()
    m4['Частка_за_кількістю_%'] = (m4['Кількість'] / len(df) * 100).round(2)
    m4['Частка_за_вартістю_%'] = (m4['Сумарна_вартість_UAH'] / df['price_uah'].sum() * 100).round(2)
    m4 = m4.sort_values(by='Кількість', ascending=False)
    print("\n[BI-4] Частка категорій у загальному каталозі:")
    print(m4.to_string(index=False))

    # 5. Найдорожчі товари у кожній категорії
    idx_max = df.groupby('category')['price_uah'].idxmax()
    m5 = df.loc[idx_max, ['category', 'product_name', 'price_uah', 'price_usd', 'price_eur', 'rating']]
    m5 = m5.sort_values(by='price_uah', ascending=False)
    print("\n[BI-5] Топ-1 найдорожчий товар у кожній категорії:")
    print(m5.to_string(index=False))

    return m1, m2, m3, m4, m5


# ==============================================================================
# 5. ВІЗУАЛІЗАЦІЯ РЕЗУЛЬТАТІВ
# ==============================================================================
def run_visualizations(df: pd.DataFrame):
    """
    Побудова та збереження аналітичних графіків у папку plots/.
    """
    print("\n" + "=" * 70)
    print("ЕТАП 5. Візуалізація результатів аналітики")
    print("=" * 70)

    sns.set_theme(style='whitegrid', font='DejaVu Sans')

    # Графік 1: Середня та медіанна ціна за категоріями
    cat_stats = df.groupby('category')['price_uah'].agg(['mean', 'median']).reset_index()
    cat_stats = cat_stats.sort_values(by='mean', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    y_pos = np.arange(len(cat_stats))
    bar_height = 0.38

    bars1 = ax.barh(y_pos + bar_height/2, cat_stats['mean'], height=bar_height,
                    label='Середня ціна (Mean)', color='#1f77b4', alpha=0.9)
    bars2 = ax.barh(y_pos - bar_height/2, cat_stats['median'], height=bar_height,
                    label='Медіанна ціна (Median)', color='#ff7f0e', alpha=0.9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(cat_stats['category'], fontsize=11)
    ax.set_xlabel('Ціна товару (UAH, ₴)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title('Рис. 1. Порівняння середньої та медіанної вартості товарів за категоріями',
                 fontsize=13, fontweight='bold', pad=15)
    ax.legend(loc='lower right', frameon=True, fontsize=10)
    ax.grid(axis='x', linestyle='--', alpha=0.6)

    for bar in bars1:
        w = bar.get_width()
        ax.text(w + 25, bar.get_y() + bar.get_height()/2, f'{w:.1f} ₴',
                va='center', ha='left', fontsize=8.5, color='#1f77b4', fontweight='bold')
    for bar in bars2:
        w = bar.get_width()
        ax.text(w + 25, bar.get_y() + bar.get_height()/2, f'{w:.1f} ₴',
                va='center', ha='left', fontsize=8.5, color='#d95f02', fontweight='bold')

    ax.set_xlim(0, cat_stats['mean'].max() * 1.18)
    plt.tight_layout()
    p1 = os.path.join(PLOTS_DIR, 'fig1_price_by_category.png')
    plt.savefig(p1)
    plt.close()
    print(f"[VIS] Збережено: {p1}")

    # Графік 2: Структура товарів за ціновими діапазонами та варіативність
    bins = [0, 1500, 2500, float('inf')]
    labels = ['Бюджетний (< 1500 ₴)', 'Середній (1500–2500 ₴)', 'Преміум (> 2500 ₴)']
    tier_counts = pd.cut(df['price_uah'], bins=bins, labels=labels).value_counts()[labels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)
    colors = ['#2ca02c', '#1f77b4', '#d62728']
    wedges, texts, autotexts = ax1.pie(
        tier_counts, labels=tier_counts.index, autopct='%1.1f%%',
        startangle=140, colors=colors, pctdistance=0.75,
        wedgeprops=dict(width=0.45, edgecolor='white', linewidth=2)
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_fontweight('bold')
    ax1.set_title('Частка товарів за ціновими сегментами', fontsize=12, fontweight='bold')

    top_cats = df['category'].value_counts().head(6).index
    df_top = df[df['category'].isin(top_cats)].copy()
    sns.boxplot(
        data=df_top, x='price_uah', y='category', hue='category',
        palette='Set2', ax=ax2, orient='h', width=0.5, legend=False,
        boxprops=dict(alpha=0.85)
    )
    ax2.set_xlabel('Ціна товару (UAH, ₴)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Категорія', fontsize=11, fontweight='bold')
    ax2.set_title('Варіативність цін у топ-6 категоріях каталогу', fontsize=12, fontweight='bold')
    ax2.grid(axis='x', linestyle='--', alpha=0.6)

    fig.suptitle('Рис. 2. Структура цінових діапазонів та дисперсія вартості товарів',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    p2 = os.path.join(PLOTS_DIR, 'fig2_price_ranges.png')
    plt.savefig(p2, bbox_inches='tight')
    plt.close()
    print(f"[VIS] Збережено: {p2}")

    # Графік 3: Співвідношення min, mean, max ціни
    m_agg = df.groupby('category')['price_uah'].agg(['min', 'mean', 'max']).reset_index()
    m_agg = m_agg.sort_values(by='mean', ascending=False)

    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    y_pos = np.arange(len(m_agg))

    for idx, row in m_agg.iterrows():
        y = list(m_agg['category']).index(row['category'])
        ax.plot([row['min'], row['max']], [y, y], color='#888888', linewidth=2.5, zorder=1)

    ax.scatter(m_agg['min'], y_pos, color='#2ca02c', s=70, label='Мінімальна ціна (Min)', zorder=2)
    ax.scatter(m_agg['mean'], y_pos, color='#1f77b4', s=90, label='Середня ціна (Mean)', zorder=2)
    ax.scatter(m_agg['max'], y_pos, color='#d62728', s=70, label='Максимальна ціна (Max)', zorder=2)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(m_agg['category'], fontsize=11)
    ax.set_xlabel('Ціна товару (UAH, ₴)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title('Рис. 3. Співвідношення мінімальної, середньої та максимальної ціни за категоріями',
                 fontsize=13, fontweight='bold', pad=15)
    ax.legend(loc='lower right', frameon=True, fontsize=10)
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    ax.set_xlim(400, 4000)

    plt.tight_layout()
    p3 = os.path.join(PLOTS_DIR, 'fig3_min_max_avg.png')
    plt.savefig(p3)
    plt.close()
    print(f"[VIS] Збережено: {p3}")


# ==============================================================================
# ГОЛОВНА ТОЧКА ВХОДУ
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="ETL-конвеєр аналітики товарів та курсів валют")
    parser.add_argument('--fetch', action='store_true', help='Примусово виконати мережевий запит до веб-сайту та API')
    args = parser.parse_args()

    print("=" * 70)
    print("РОЗПОЧАТО ВИКОНАННЯ ETL-КОНВЕЄРА (ЛАБОРАТОРНА РОБОТА №6)")
    print("=" * 70)

    # 1. Extract
    df_raw, rates = run_extract(force_fetch=args.fetch)

    # 2. Transform
    df_processed = run_transform(df_raw, rates)

    # 3. Load & Benchmark
    bench_results = run_load_and_benchmark(df_processed)

    # 4. BI Analytics
    m1, m2, m3, m4, m5 = run_bi_analytics(df_processed)

    # 5. Visualizations
    run_visualizations(df_processed)

    print("\n" + "=" * 70)
    print("ETL-КОНВЕЄР УСПІШНО ЗАВЕРШЕНО!")
    print("=" * 70)


if __name__ == '__main__':
    main()
