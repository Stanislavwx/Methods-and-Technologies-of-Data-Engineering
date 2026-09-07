"""
Інтерактивний аналітичний дашборд показників E-Commerce (Superstore Dataset)
Лабораторна робота №2: Візуалізація даних та побудова аналітичних дашбордів
"""

import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# 1. Завантаження та базова підготовка даних
def load_data(filepath="superstore.csv"):
    df = pd.read_csv(filepath)
    # Видалення порожніх рядків
    df = df.dropna(subset=['Order ID', 'Order Date', 'Customer ID']).copy()
    # Імпутація відсутнього поштового індексу (Burlington, Vermont)
    df.loc[(df['City'] == 'Burlington') & (df['State'] == 'Vermont') & (df['Postal Code'].isnull()), 'Postal Code'] = 5401
    df['Postal Code'] = df['Postal Code'].astype(int).astype(str).str.zfill(5)
    # Форматування дат та створення аналітичних ознак
    df['Order Date'] = pd.to_datetime(df['Order Date'], format='mixed')
    df['Ship Date'] = pd.to_datetime(df['Ship Date'], format='mixed')
    df['Order_YearMonth'] = df['Order Date'].dt.to_period('M').astype(str)
    df['Discount_Pct'] = df['Discount'] * 100
    df['Profit_Margin_Pct'] = (df['Profit'] / df['Sales']) * 100
    return df

df = load_data()

# Довідник кодів штатів США для картограми
US_STATE_CODES = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
    'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA',
    'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA',
    'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
    'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS', 'Missouri': 'MO',
    'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ',
    'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH',
    'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
    'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT',
    'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY',
    'District of Columbia': 'DC'
}

# Ініціалізація Dash додатку
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY, dbc.icons.BOOTSTRAP],
    title="E-Commerce Analytics Dashboard | Lab 2"
)
server = app.server

# Компонент KPI-картки
def create_kpi_card(title, value_id, icon_class, color_hex):
    return dbc.Card(
        dbc.CardBody([
            html.Div([
                html.I(className=f"bi {icon_class} fs-2 me-3", style={"color": color_hex}),
                html.Div([
                    html.H6(title, className="text-muted mb-1 text-uppercase", style={"fontSize": "0.75rem", "letterSpacing": "0.5px"}),
                    html.H3(id=value_id, className="fw-bold mb-0 text-dark", style={"fontSize": "1.45rem"}),
                ])
            ], className="d-flex align-items-center")
        ]),
        className="shadow-sm border-0 mb-3",
        style={"borderLeft": f"4px solid {color_hex}", "borderRadius": "8px", "background": "#FFFFFF"}
    )

# Панель фільтрів
filter_panel = dbc.Card(
    dbc.CardBody([
        html.H5([html.I(className="bi bi-funnel-fill me-2 text-primary"), "Фільтри даних"], className="card-title mb-3 fw-bold"),
        
        html.Label("Період замовлення:", className="fw-bold text-secondary mb-1 small"),
        dcc.DatePickerRange(
            id="date-picker-range",
            min_date_allowed=df['Order Date'].min().date(),
            max_date_allowed=df['Order Date'].max().date(),
            start_date=df['Order Date'].min().date(),
            end_date=df['Order Date'].max().date(),
            display_format="YYYY-MM-DD",
            className="mb-3 w-100",
            style={"fontSize": "0.8rem"}
        ),
        
        html.Label("Регіон:", className="fw-bold text-secondary mb-1 small"),
        dcc.Dropdown(
            id="region-dropdown",
            options=[{"label": "Всі регіони (All)", "value": "All"}] + [
                {"label": r, "value": r} for r in sorted(df['Region'].unique())
            ],
            value="All",
            clearable=False,
            className="mb-3"
        ),
        
        html.Label("Категорія товару:", className="fw-bold text-secondary mb-1 small"),
        dcc.Dropdown(
            id="category-dropdown",
            options=[{"label": "Всі категорії (All)", "value": "All"}] + [
                {"label": c, "value": c} for c in sorted(df['Category'].unique())
            ],
            value="All",
            clearable=False,
            className="mb-3"
        ),
        
        html.Label("Показник для карти:", className="fw-bold text-secondary mb-1 small"),
        dbc.RadioItems(
            id="map-metric-radio",
            options=[
                {"label": "Виручка (Sales)", "value": "Sales"},
                {"label": "Прибуток (Profit)", "value": "Profit"}
            ],
            value="Sales",
            inline=True,
            className="mb-3 text-secondary"
        ),
        
        dbc.Button([html.I(className="bi bi-arrow-counterclockwise me-2"), "Скинути фільтри"], id="reset-btn", color="outline-secondary", size="sm", className="w-100 mt-2")
    ]),
    className="shadow-sm border-0 mb-4",
    style={"borderRadius": "10px", "backgroundColor": "#F8F9FA"}
)

# Макет сторінки
app.layout = dbc.Container([
    # Заголовок
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Div([
                    html.H2("Аналітичний дашборд E-Commerce (Superstore)", className="fw-bold text-white mb-1"),
                    html.P("Лабораторна робота №2: Візуалізація даних та побудова аналітичних дашбордів | Інженерія даних", className="text-light mb-0")
                ]),
                html.Div([
                    dbc.Badge("Plotly Dash", color="light", text_color="dark", className="me-2 p-2"),
                    dbc.Badge("9,994 Транзакції", color="success", className="p-2")
                ], className="d-none d-md-flex align-items-center")
            ], className="d-flex justify-content-between align-items-center p-4 rounded-3 shadow-sm mb-4", style={"backgroundColor": "#2C3E50"})
        ], width=12)
    ]),
    
    # KPI картки
    dbc.Row([
        dbc.Col(create_kpi_card("Загальний виторг", "kpi-sales", "bi-cash-stack", "#3498DB"), xs=12, sm=6, lg=True),
        dbc.Col(create_kpi_card("Чистий прибуток", "kpi-profit", "bi-graph-up-arrow", "#18BC9C"), xs=12, sm=6, lg=True),
        dbc.Col(create_kpi_card("Рентабельність (Margin)", "kpi-margin", "bi-percent", "#F39C12"), xs=12, sm=6, lg=True),
        dbc.Col(create_kpi_card("Кількість замовлень", "kpi-orders", "bi-bag-check-fill", "#9B59B6"), xs=12, sm=6, lg=True),
        dbc.Col(create_kpi_card("Середній чек (AOV)", "kpi-aov", "bi-receipt-cutoff", "#2C3E50"), xs=12, sm=6, lg=True),
    ], className="mb-2"),
    
    # Основний контент (Фільтри + 4 візуалізації)
    dbc.Row([
        # Лівий сайдбар фільтрів
        dbc.Col([filter_panel], xs=12, lg=3),
        
        # Візуалізації
        dbc.Col([
            # Рядок 1: Динаміка у часі + Treemap асортименту
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.I(className="bi bi-clock-history me-2 text-primary"),
                            html.Strong("1. Динаміка продажів та прибутку у часі (Line & Area)")
                        ], className="bg-white border-0 pt-3 pb-0"),
                        dbc.CardBody([dcc.Graph(id="trend-chart", style={"height": "350px"}, config={"displayModeBar": False})])
                    ], className="shadow-sm border-0 mb-4", style={"borderRadius": "10px"})
                ], xs=12, xl=7),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.I(className="bi bi-diagram-3-fill me-2 text-primary"),
                            html.Strong("2. Ієрархія товарів: Категорії та Прибутковість (Treemap)")
                        ], className="bg-white border-0 pt-3 pb-0"),
                        dbc.CardBody([dcc.Graph(id="treemap-chart", style={"height": "350px"}, config={"displayModeBar": False})])
                    ], className="shadow-sm border-0 mb-4", style={"borderRadius": "10px"})
                ], xs=12, xl=5)
            ]),
            
            # Рядок 2: Географічна карта + Скаттер дисконту/прибутку
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.I(className="bi bi-geo-alt-fill me-2 text-primary"),
                            html.Strong("3. Географічний розподіл по штатах США (Choropleth Map)")
                        ], className="bg-white border-0 pt-3 pb-0"),
                        dbc.CardBody([dcc.Graph(id="geo-map", style={"height": "380px"}, config={"displayModeBar": False})])
                    ], className="shadow-sm border-0 mb-4", style={"borderRadius": "10px"})
                ], xs=12, xl=7),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.I(className="bi bi-scatter me-2 text-primary"),
                            html.Strong("4. Вплив знижок на прибуток (Scatter Plot з порогом)")
                        ], className="bg-white border-0 pt-3 pb-0"),
                        dbc.CardBody([dcc.Graph(id="scatter-chart", style={"height": "380px"}, config={"displayModeBar": False})])
                    ], className="shadow-sm border-0 mb-4", style={"borderRadius": "10px"})
                ], xs=12, xl=5)
            ])
        ], xs=12, lg=9)
    ]),
    
    # Підвал
    dbc.Row([
        dbc.Col([
            html.Hr(),
            html.P("© Лабораторна робота №2 з дисципліни 'Інженерія даних'", className="text-center text-muted small mb-3")
        ], width=12)
    ])
], fluid=True, className="p-4 bg-light")

# Колбек інтерактивності
@app.callback(
    [
        Output("kpi-sales", "children"),
        Output("kpi-profit", "children"),
        Output("kpi-margin", "children"),
        Output("kpi-orders", "children"),
        Output("kpi-aov", "children"),
        Output("trend-chart", "figure"),
        Output("treemap-chart", "figure"),
        Output("geo-map", "figure"),
        Output("scatter-chart", "figure"),
    ],
    [
        Input("date-picker-range", "start_date"),
        Input("date-picker-range", "end_date"),
        Input("region-dropdown", "value"),
        Input("category-dropdown", "value"),
        Input("map-metric-radio", "value"),
        Input("reset-btn", "n_clicks"),
    ]
)
def update_dashboard(start_date, end_date, selected_region, selected_category, map_metric, reset_clicks):
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None

    if triggered_id == "reset-btn":
        filtered_df = df.copy()
    else:
        filtered_df = df.copy()
        if start_date:
            filtered_df = filtered_df[filtered_df['Order Date'] >= pd.to_datetime(start_date)]
        if end_date:
            filtered_df = filtered_df[filtered_df['Order Date'] <= pd.to_datetime(end_date)]
        if selected_region and selected_region != "All":
            filtered_df = filtered_df[filtered_df['Region'] == selected_region]
        if selected_category and selected_category != "All":
            filtered_df = filtered_df[filtered_df['Category'] == selected_category]

    if len(filtered_df) == 0:
        filtered_df = df.head(1).copy()

    # Розрахунок KPI
    total_sales = filtered_df['Sales'].sum()
    total_profit = filtered_df['Profit'].sum()
    margin_pct = (total_profit / total_sales * 100) if total_sales > 0 else 0
    total_orders = filtered_df['Order ID'].nunique()
    aov = (total_sales / total_orders) if total_orders > 0 else 0

    # 1. Графік тренду (Line & Area)
    monthly = filtered_df.groupby('Order_YearMonth').agg(
        Sales=('Sales', 'sum'),
        Profit=('Profit', 'sum')
    ).reset_index()

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=monthly['Order_YearMonth'], y=monthly['Sales'],
        name="Виручка ($)", mode='lines', fill='tozeroy',
        line=dict(color="#3498DB", width=2.5),
        fillcolor='rgba(52, 152, 219, 0.15)',
        hovertemplate="Місяць: %{x}<br>Виручка: $%{y:,.0f}<extra></extra>"
    ))
    fig_trend.add_trace(go.Scatter(
        x=monthly['Order_YearMonth'], y=monthly['Profit'],
        name="Прибуток ($)", mode='lines+markers',
        line=dict(color="#18BC9C", width=2),
        marker=dict(size=4),
        hovertemplate="Місяць: %{x}<br>Прибуток: $%{y:,.0f}<extra></extra>"
    ))
    fig_trend.update_layout(
        margin=dict(l=40, r=20, t=20, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="white",
        xaxis=dict(showgrid=True, gridcolor="#ECEFF1", tickangle=-45),
        yaxis=dict(showgrid=True, gridcolor="#ECEFF1", title="USD")
    )

    # 2. Treemap ієрархії товарів
    tree_data = filtered_df.groupby(['Category', 'Sub-Category']).agg(
        Sales=('Sales', 'sum'),
        Profit=('Profit', 'sum')
    ).reset_index()
    tree_data['Margin_Pct'] = (tree_data['Profit'] / tree_data['Sales'] * 100).round(1)

    fig_tree = px.treemap(
        tree_data,
        path=[px.Constant("Всі категорії"), 'Category', 'Sub-Category'],
        values='Sales',
        color='Margin_Pct',
        color_continuous_scale=['#E74C3C', '#F39C12', '#2ECC71'],
        color_continuous_midpoint=0,
        hover_data={'Sales': ':.0f', 'Profit': ':.0f', 'Margin_Pct': ':.1f'}
    )
    fig_tree.update_layout(
        margin=dict(l=10, r=10, t=20, b=10),
        coloraxis_colorbar=dict(title="Маржа %", len=0.7)
    )

    # 3. Гео-карта по штатах
    state_agg = filtered_df.groupby('State').agg(
        Sales=('Sales', 'sum'),
        Profit=('Profit', 'sum'),
        Orders=('Order ID', 'nunique')
    ).reset_index()
    state_agg['State_Code'] = state_agg['State'].map(US_STATE_CODES)
    state_agg['Margin_Pct'] = (state_agg['Profit'] / state_agg['Sales'] * 100).round(1)

    color_scale = 'Blues' if map_metric == 'Sales' else ['#E74C3C', '#ECEFF1', '#2ECC71']
    midpoint = None if map_metric == 'Sales' else 0

    fig_map = px.choropleth(
        state_agg,
        locations='State_Code',
        locationmode="USA-states",
        color=map_metric,
        scope="usa",
        color_continuous_scale=color_scale,
        color_continuous_midpoint=midpoint,
        hover_name='State',
        hover_data={'Sales': ':$,.0f', 'Profit': ':$,.0f', 'Orders': ':,', 'Margin_Pct': ':.1f%'}
    )
    fig_map.update_layout(
        geo=dict(bgcolor='rgba(0,0,0,0)', lakecolor='#F0F3F4'),
        margin=dict(l=0, r=0, t=10, b=0),
        coloraxis_colorbar=dict(title="USD", len=0.7)
    )

    # 4. Скаттер знижка vs прибуток
    sample_size = min(1200, len(filtered_df))
    scatter_df = filtered_df.sample(sample_size, random_state=42) if len(filtered_df) > sample_size else filtered_df
    fig_scatter = px.scatter(
        scatter_df,
        x="Discount_Pct", y="Profit", size="Sales", color="Category",
        color_discrete_map={"Technology": "#3498DB", "Furniture": "#E74C3C", "Office Supplies": "#2ECC71"},
        hover_name="Sub-Category",
        hover_data={"Sales": ':$,.2f', "Profit": ':$,.2f', "Discount_Pct": ':.1f%'},
        size_max=30, opacity=0.6
    )
    fig_scatter.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig_scatter.add_vline(x=20, line_dash="dot", line_color="red", annotation_text="Поріг 20% знижки", annotation_position="top right")
    fig_scatter.update_layout(
        margin=dict(l=40, r=20, t=20, b=40),
        plot_bgcolor="white",
        xaxis=dict(showgrid=True, gridcolor="#ECEFF1", title="Знижка (%)"),
        yaxis=dict(showgrid=True, gridcolor="#ECEFF1", title="Прибуток ($)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return (
        f"${total_sales:,.0f}",
        f"${total_profit:,.0f}",
        f"{margin_pct:.1f}%",
        f"{total_orders:,}",
        f"${aov:,.2f}",
        fig_trend,
        fig_tree,
        fig_map,
        fig_scatter
    )

if __name__ == '__main__':
    print("Запуск Dash сервера на порту 8050...")
    app.run(debug=False, host='0.0.0.0', port=8050)
