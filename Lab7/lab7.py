import os
import time
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Налаштування стилю візуалізацій
sns.set_theme(style="whitegrid", font="DejaVu Sans")
os.makedirs("plots", exist_ok=True)


def load_and_preprocess_data(filepath="data_student_rfm1.csv"):
    """
    Завдання 1. Первинна підготовка даних
    Завантаження, перевірка розмірності, типів, пропусків, дублікатів та створення amount.
    """
    print("=" * 60)
    print("1. ПЕРВИННА ПІДГОТОВКА ДАНИХ")
    print("=" * 60)

    df_raw = pd.read_csv(filepath)
    if "Unnamed: 0" in df_raw.columns:
        df_raw = df_raw.drop(columns=["Unnamed: 0"])

    print(f"Початковий розмір: {df_raw.shape[0]} рядків, {df_raw.shape[1]} колонок")
    print("\nТипи даних початкових ознак:")
    print(df_raw.dtypes)

    print(f"\nКількість унікальних інвойсів (транзакцій): {df_raw['InvoiceNo'].nunique()}")
    print(f"Кількість унікальних покупців (включно з NaN): {df_raw['CustomerID'].nunique(dropna=False)}")
    print(f"Кількість унікальних валідних покупців: {df_raw['CustomerID'].nunique(dropna=True)}")

    print("\nПропущені значення за стовпцями:")
    print(df_raw.isnull().sum())

    # Вилучення записів без CustomerID, оскільки агрегація проводиться за клієнтами
    df_clean = df_raw.dropna(subset=["CustomerID"]).copy()

    # Перевірка та видалення дублікатів
    dup_count = df_clean.duplicated().sum()
    print(f"\nВиявлено повних дублікатів рядків: {dup_count}")
    if dup_count > 0:
        df_clean = df_clean.drop_duplicates().copy()

    # Приведення типів даних
    df_clean["customer_id"] = df_clean["CustomerID"].astype(int)
    df_clean["purchase_date"] = pd.to_datetime(df_clean["InvoiceDate"])
    df_clean["quantity"] = df_clean["Quantity"].astype(int)
    df_clean["unit_price"] = df_clean["UnitPrice"].astype(float)

    # Розрахунок вартості покупки
    df_clean["amount"] = df_clean["quantity"] * df_clean["unit_price"]

    print(f"\nРозмір після очищення: {df_clean.shape[0]} транзакцій, {df_clean['customer_id'].nunique()} унікальних клієнтів")
    print(f"Діапазон дат: {df_clean['purchase_date'].min()} — {df_clean['purchase_date'].max()}")

    return df_clean


def split_temporal_batches(df):
    """
    Завдання 2. Формування часових пакетів даних
    Сортування за purchase_date та поділ на три неперетинні періоди.
    """
    print("\n" + "=" * 60)
    print("2. ФОРМУВАННЯ ЧАСОВИХ ПАКЕТІВ ДАНИХ")
    print("=" * 60)

    df_sorted = df.sort_values("purchase_date").reset_index(drop=True)

    min_date = df_sorted["purchase_date"].min()
    max_date = df_sorted["purchase_date"].max()
    delta = (max_date - min_date) / 3
    t1 = min_date + delta
    t2 = min_date + 2 * delta

    p1 = df_sorted[df_sorted["purchase_date"] < t1].copy()
    p2 = df_sorted[(df_sorted["purchase_date"] >= t1) & (df_sorted["purchase_date"] < t2)].copy()
    p3 = df_sorted[df_sorted["purchase_date"] >= t2].copy()

    batches = [
        ("Period 1", "transactions_period_1.csv", p1),
        ("Period 2", "transactions_period_2.csv", p2),
        ("Period 3", "transactions_period_3.csv", p3),
    ]

    for name, filename, b_df in batches:
        b_df.to_csv(filename, index=False)
        print(f"Збережено {filename}:")
        print(f"  Діапазон: {b_df['purchase_date'].min()} — {b_df['purchase_date'].max()}")
        print(f"  Кількість транзакцій: {len(b_df)}, унікальних покупців: {b_df['customer_id'].nunique()}")

    # Контрольна перевірка відсутності перетинів
    assert p1["purchase_date"].max() < p2["purchase_date"].min(), "Помилка: перетин дат між 1 і 2 періодами!"
    assert p2["purchase_date"].max() < p3["purchase_date"].min(), "Помилка: перетин дат між 2 і 3 періодами!"
    print("Контрольна перевірка: періоди строго послідовні та не перетинаються за датами.")

    return batches


def aggregate_batch(batch_df):
    """
    Розрахунок зведених метрик для кожного покупця:
    - customer_id: ідентифікатор покупця
    - number_of_purchases: кількість покупок (рядків транзакцій)
    - total_quantity: загальна кількість одиниць товару
    - total_spent: загальна вартість покупок
    - last_purchase_date: дата останньої покупки
    """
    agg = batch_df.groupby("customer_id").agg(
        number_of_purchases=("quantity", "count"),
        total_quantity=("quantity", "sum"),
        total_spent=("amount", "sum"),
        last_purchase_date=("purchase_date", "max"),
    ).reset_index()
    return agg


def write_log(period_num, filename, b_df, new_count, updated_count, total_count, log_file="processing.log"):
    """
    Завдання 6. Ведення журналу обробки processing.log
    """
    mode = "w" if period_num == 1 else "a"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    entry = (
        f"Period {period_num}\n"
        f"File: {filename}\n"
        f"Date range: {b_df['purchase_date'].min()} - {b_df['purchase_date'].max()}\n"
        f"Transactions processed: {len(b_df)}\n"
        f"Customers in batch: {b_df['customer_id'].nunique()}\n"
        f"New customers: {new_count}\n"
        f"Updated customers: {updated_count}\n"
        f"Customers after processing: {total_count}\n"
        f"Execution time: {timestamp}\n"
        f"Status: SUCCESS\n"
        f"{'-' * 50}\n"
    )

    with open(log_file, mode, encoding="utf-8") as f:
        f.write(entry)


def run_incremental_pipeline(batches):
    """
    Завдання 3, 4, 5. Initial Load та інкрементальна обробка пакетів 2 і 3.
    """
    print("\n" + "=" * 60)
    print("3-5. ІНКРЕМЕНТАЛЬНА ОБРОБКА ТА ВЕДЕННЯ ЖУРНАЛУ")
    print("=" * 60)

    customers_current = None
    summary_records = []

    for idx, (p_name, filename, b_df) in enumerate(batches, start=1):
        # Ізольоване завантаження чергового пакета з диска
        batch_data = pd.read_csv(filename)
        batch_data["purchase_date"] = pd.to_datetime(batch_data["purchase_date"])
        batch_agg = aggregate_batch(batch_data).set_index("customer_id")

        if idx == 1:
            # Завдання 3: Initial Load
            customers_current = batch_agg.copy()
            new_cust_count = len(customers_current)
            updated_cust_count = 0
            print(f"[{p_name}] Initial Load виконано: додано {new_cust_count} нових клієнтів.")
        else:
            # Завдання 4 та 5: Incremental Load (INSERT / UPDATE)
            current_ids = set(customers_current.index)
            batch_ids = set(batch_agg.index)

            new_ids = list(batch_ids - current_ids)
            common_ids = list(batch_ids & current_ids)

            # UPDATE для існуючих клієнтів
            if common_ids:
                customers_current.loc[common_ids, "number_of_purchases"] += batch_agg.loc[common_ids, "number_of_purchases"]
                customers_current.loc[common_ids, "total_quantity"] += batch_agg.loc[common_ids, "total_quantity"]
                customers_current.loc[common_ids, "total_spent"] += batch_agg.loc[common_ids, "total_spent"]
                # Оновлення дати останньої покупки: максимум(попередня, нова)
                customers_current.loc[common_ids, "last_purchase_date"] = np.maximum(
                    pd.to_datetime(customers_current.loc[common_ids, "last_purchase_date"]),
                    pd.to_datetime(batch_agg.loc[common_ids, "last_purchase_date"]),
                )

            # INSERT для нових клієнтів
            if new_ids:
                new_records = batch_agg.loc[new_ids]
                customers_current = pd.concat([customers_current, new_records])

            new_cust_count = len(new_ids)
            updated_cust_count = len(common_ids)
            print(f"[{p_name}] Інкрементальне оновлення: {new_cust_count} нових (INSERT), {updated_cust_count} оновлених (UPDATE).")

        total_customers = len(customers_current)
        # Збереження оновленої таблиці клієнтів
        customers_current.sort_index().reset_index().to_csv("customers_current.csv", index=False)

        # Запис події в журнал
        write_log(idx, filename, batch_data, new_cust_count, updated_cust_count, total_customers)

        summary_records.append({
            "Період": p_name,
            "Файл": filename,
            "Транзакцій": len(batch_data),
            "Клієнтів у пакеті": batch_data["customer_id"].nunique(),
            "Нових покупців": new_cust_count,
            "Оновлено покупців": updated_cust_count,
            "Покупців після обробки": total_customers,
        })

    customers_incremental = customers_current.sort_index().reset_index()
    customers_incremental.to_csv("customers_incremental.csv", index=False)
    summary_df = pd.DataFrame(summary_records)

    return customers_incremental, summary_df


def run_full_processing(df):
    """
    Завдання 7. Контрольна повна обробка набору даних.
    """
    print("\n" + "=" * 60)
    print("7. ПЕРЕВІРКА ПОВНОЇ ТА ІНКРЕМЕНТАЛЬНОЇ ОБРОБКИ")
    print("=" * 60)

    customers_full = aggregate_batch(df).sort_values("customer_id").reset_index(drop=True)
    customers_full.to_csv("customers_full.csv", index=False)
    print(f"Повна обробка завершена: {len(df)} транзакцій, {len(customers_full)} клієнтів.")
    return customers_full


def verify_results(customers_full, customers_incremental):
    """
    Звірка результатів: customers_full vs customers_incremental.
    """
    customers_inc = customers_incremental.sort_values("customer_id").reset_index(drop=True)
    customers_fl = customers_full.sort_values("customer_id").reset_index(drop=True)

    len_match = len(customers_fl) == len(customers_inc)
    id_diff = (customers_fl["customer_id"] != customers_inc["customer_id"]).sum()
    purch_diff = (customers_fl["number_of_purchases"] != customers_inc["number_of_purchases"]).sum()
    qty_diff = (customers_fl["total_quantity"] != customers_inc["total_quantity"]).sum()
    spent_diff = (~np.isclose(customers_fl["total_spent"], customers_inc["total_spent"], atol=1e-5)).sum()
    date_diff = (pd.to_datetime(customers_fl["last_purchase_date"]) != pd.to_datetime(customers_inc["last_purchase_date"])).sum()

    print("\n--- РЕЗУЛЬТАТИ ВЕРИФІКАЦІЇ ---")
    print(f"1. Кількість покупців збігається: {len_match} ({len(customers_fl)} == {len(customers_inc)})")
    print(f"2. Розбіжностей у customer_id: {id_diff}")
    print(f"3. Розбіжностей у number_of_purchases: {purch_diff}")
    print(f"4. Розбіжностей у total_quantity: {qty_diff}")
    print(f"5. Розбіжностей у total_spent: {spent_diff}")
    print(f"6. Розбіжностей у last_purchase_date: {date_diff}")

    all_matched = (len_match and id_diff == 0 and purch_diff == 0 and qty_diff == 0 and spent_diff == 0 and date_diff == 0)
    if all_matched:
        print("\n>>> ПЕРЕВІРКА ПРОЙШЛА УСПІШНО: РЕЗУЛЬТАТИ 100% ІДЕНТИЧНІ! <<<")
    else:
        print("\n>>> ПОМИЛКА: Знайдено розбіжності! <<<")

    return all_matched


def generate_visualizations(summary_df, df_clean):
    """
    Завдання 8. Побудова аналітичних графіків.
    """
    print("\n" + "=" * 60)
    print("8. ПОБУДОВА ВІЗУАЛІЗАЦІЙ")
    print("=" * 60)

    # Рис. 1: Динаміка нових та оновлених клієнтів
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(summary_df))
    width = 0.35

    ax.bar(x - width/2, summary_df["Нових покупців"], width, label="Нові покупці (INSERT)", color="#2b5c8f")
    ax.bar(x + width/2, summary_df["Оновлено покупців"], width, label="Оновлені покупці (UPDATE)", color="#4ca64c")

    ax.set_ylabel("Кількість клієнтів", fontsize=12, color="black")
    ax.set_title("Динаміка сегментації клієнтів за пакетами обробки", fontsize=13, fontweight="bold", color="black")
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["Період"], fontsize=11, color="black")
    ax.tick_params(colors="black")
    ax.legend(frameon=True)

    for i in x:
        ax.annotate(str(summary_df.loc[i, "Нових покупців"]), (i - width/2, summary_df.loc[i, "Нових покупців"] + 30),
                    ha="center", fontsize=10, fontweight="bold", color="black")
        if summary_df.loc[i, "Оновлено покупців"] > 0:
            ax.annotate(str(summary_df.loc[i, "Оновлено покупців"]), (i + width/2, summary_df.loc[i, "Оновлено покупців"] + 30),
                        ha="center", fontsize=10, fontweight="bold", color="black")

    plt.tight_layout()
    fig1_path = "plots/fig1_customer_dynamics.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"Збережено графік: {fig1_path}")

    # Рис. 2: Накопичувальне зростання клієнтської бази та обсяг пакетів
    fig, ax1 = plt.subplots(figsize=(8, 5))
    color = "#2b5c8f"
    ax1.set_xlabel("Період обробки", fontsize=12, color="black")
    ax1.set_ylabel("Накопичена база клієнтів (чол.)", color=color, fontsize=12)
    line1 = ax1.plot(summary_df["Період"], summary_df["Покупців після обробки"], marker="o", color=color, linewidth=2.5, label="Клієнтів у базі")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.tick_params(colors="black")

    for i, txt in enumerate(summary_df["Покупців після обробки"]):
        ax1.annotate(str(txt), (summary_df.loc[i, "Період"], txt + 70), ha="center", fontsize=10, fontweight="bold", color=color)

    ax2 = ax1.twinx()
    color2 = "#d95f02"
    ax2.set_ylabel("Транзакцій у пакеті (од.)", color=color2, fontsize=12)
    line2 = ax2.plot(summary_df["Період"], summary_df["Транзакцій"], marker="s", color=color2, linewidth=2, linestyle="--", label="Транзакцій у пакеті")
    ax2.tick_params(axis="y", labelcolor=color2)

    for i, txt in enumerate(summary_df["Транзакцій"]):
        ax2.annotate(str(txt), (summary_df.loc[i, "Період"], txt + 500), ha="center", fontsize=10, color=color2)

    plt.title("Накопичувальне зростання клієнтської бази та обсяг пакетів", fontsize=13, fontweight="bold", color="black")
    plt.tight_layout()
    fig2_path = "plots/fig2_cumulative_customers.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"Збережено графік: {fig2_path}")

    # Рис. 3: Бенчмаркінг обсягу сканування (Full vs Incremental)
    t1_len = summary_df.loc[0, "Транзакцій"]
    t2_len = summary_df.loc[1, "Транзакцій"]
    t3_len = summary_df.loc[2, "Транзакцій"]

    full_scans = [t1_len, t1_len + t2_len, t1_len + t2_len + t3_len]
    inc_scans = [t1_len, t2_len, t3_len]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(summary_df))
    width = 0.35

    ax.bar(x - width/2, full_scans, width, label="Full Processing (повторне сканування)", color="#c0392b")
    ax.bar(x + width/2, inc_scans, width, label="Incremental Processing (тільки дельта)", color="#27ae60")

    ax.set_ylabel("Кількість оброблених рядків", fontsize=12, color="black")
    ax.set_title("Порівняння обсягу сканування: Full vs Incremental", fontsize=13, fontweight="bold", color="black")
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["Період"], fontsize=11, color="black")
    ax.tick_params(colors="black")
    ax.legend(frameon=True)

    for i in x:
        ax.annotate(f"{full_scans[i]:,}".replace(",", " "), (i - width/2, full_scans[i] + 700),
                    ha="center", fontsize=9, fontweight="bold", color="black")
        ax.annotate(f"{inc_scans[i]:,}".replace(",", " "), (i + width/2, inc_scans[i] + 700),
                    ha="center", fontsize=9, fontweight="bold", color="black")

    plt.tight_layout()
    fig3_path = "plots/fig3_full_vs_incremental_benchmark.png"
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"Збережено графік: {fig3_path}")


def print_summary_and_conclusion(summary_df):
    """
    Завдання 8 і 9. Підсумкова таблиця та висновки.
    """
    print("\n" + "=" * 60)
    print("8. ПІДСУМКОВА ТАБЛИЦЯ ДИНАМІКИ ЗМІН (з processing.log)")
    print("=" * 60)
    print(summary_df.to_string(index=False))

    t1 = summary_df.loc[0, "Транзакцій"]
    t2 = summary_df.loc[1, "Транзакцій"]
    t3 = summary_df.loc[2, "Транзакцій"]
    full_total = t1 + (t1 + t2) + (t1 + t2 + t3)
    inc_total = t1 + t2 + t3
    saving_pct = (1 - inc_total / full_total) * 100

    print("\n" + "=" * 60)
    print("9. ВИСНОВОК: ПЕРЕВАГИ ІНКРЕМЕНТАЛЬНОЇ ОБРОБКИ")
    print("=" * 60)
    print(
        f"Різниця між Full Processing та Incremental Processing полягає у стратегії роботи з новими даними:\n"
        f"1. Full Processing щоразу зчитує весь накопичений історичний масив даних та виконує повну повторну агрегацію.\n"
        f"   Сумарно при 3 періодах було б оброблено: {t1} + {t1 + t2} + {t1 + t2 + t3} = {full_total} рядків.\n"
        f"2. Incremental Processing обробляє виключно дельту (новий часовий пакет) та вносить точкові зміни\n"
        f"   (INSERT для нових клієнтів, UPDATE для існуючих) у сформовану таблицю стану customers_current.\n"
        f"   Сумарно оброблено рівно: {t1} + {t2} + {t3} = {inc_total} рядків (економія {saving_pct:.1f}% обчислень).\n"
        f"\n"
        f"Чому доцільніше обробляти тільки новий фрагмент:\n"
        f"• Економія I/O та пам'яті: зчитується лише новий пакет, що не навантажує дискову підсистему та RAM.\n"
        f"• Масштабованість (Scalability): при регулярному надходженні даних Full Processing демонструє квадратичне\n"
        f"  зростання складності O(N^2) накопичених операцій, тоді як Incremental має лінійну складність O(N).\n"
        f"• Швидкість (Latency): оновлення стану відбувається швидко, дозволяючи системам працювати у режимі near\n"
        f"  real-time та забезпечувати актуальну аналітику без затримок."
    )


def main():
    start_time = time.time()

    # Завдання 1
    df_clean = load_and_preprocess_data("data_student_rfm1.csv")

    # Завдання 2
    batches = split_temporal_batches(df_clean)

    # Завдання 3, 4, 5, 6
    customers_incremental, summary_df = run_incremental_pipeline(batches)

    # Завдання 7
    customers_full = run_full_processing(df_clean)
    verify_results(customers_full, customers_incremental)

    # Завдання 8
    generate_visualizations(summary_df, df_clean)

    # Завдання 8 і 9
    print_summary_and_conclusion(summary_df)

    elapsed = time.time() - start_time
    print(f"\nЗагальний час виконання: {elapsed:.2f} с.")


if __name__ == "__main__":
    main()
