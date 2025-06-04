import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from datetime import timedelta
import matplotlib.pyplot as plt  # solo para cálculos, no para graficar en Streamlit
import streamlit as st
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.graph_objects as go

# ----------------------------------------
# Configuración de página (debe ser el primer comando de Streamlit)
# ----------------------------------------
st.set_page_config(
    page_title="Dashboard de Estacionalidad y Pronóstico",
    layout="wide"
)

# ----------------------------------------
# 0. Generación de datos artificiales (sin eventos de junio)
# ----------------------------------------
@st.cache_data
def generate_data():
    end_date = pd.Timestamp.today().normalize()
    start_date = end_date - pd.Timedelta(days=5 * 365)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    categories = ['Bebidas', 'Cafés', 'Snacks', 'Jugos', 'Galletas']

    df = pd.DataFrame({
        'fecha': np.repeat(dates, len(categories)),
        'categoría': categories * len(dates)
    })
    df['mes'] = df['fecha'].dt.month
    df['factor_estacional'] = 1 + 0.3 * np.sin(2 * np.pi * (df['mes'] - 1) / 3)

    baseline = {
        'Bebidas': 500,
        'Cafés':   300,
        'Snacks':  200,
        'Jugos':   400,
        'Galletas':250
    }
    df['volumen_base'] = df['categoría'].map(baseline)

    np.random.seed(42)
    noise = np.random.normal(
        loc=0,
        scale=0.1 * df['volumen_base'],
        size=len(df)
    )
    df['ventas_diarias'] = (
        df['volumen_base'] * df['factor_estacional'] + noise
    ).round().astype(int).clip(lower=0)
    df = df[['fecha', 'categoría', 'ventas_diarias']]

    df['fecha'] = pd.to_datetime(df['fecha'])
    df = df.set_index('fecha')
    ts_daily = df.pivot_table(
        index='fecha',
        columns='categoría',
        values='ventas_diarias',
        aggfunc='sum'
    ).asfreq('D', fill_value=0)

    return ts_daily

ts_daily = generate_data()

# ----------------------------------------
# 1. Función para crear features (omitimos items 7–10)
# ----------------------------------------
def create_features(df, dropna_y=True):
    """
    Dado un DataFrame con índice datetime y una sola columna (la serie 'y'),
    devuelve un DataFrame de features para LightGBM.
    """
    df_feat = df.copy()
    df_feat.index = pd.to_datetime(df_feat.index)
    df_feat = df_feat.rename(columns={df_feat.columns[0]: 'y'})
    df_feat['dayofweek']        = df_feat.index.dayofweek
    df_feat['month']            = df_feat.index.month
    df_feat['quarter']          = df_feat.index.quarter
    df_feat['year']             = df_feat.index.year
    df_feat['dayofmonth']       = df_feat.index.day
    df_feat['is_weekend']       = (df_feat.index.dayofweek >= 5).astype(int)
    df_feat['dayofyear']        = df_feat.index.dayofyear
    df_feat['is_month_start']   = df_feat.index.is_month_start.astype(int)
    df_feat['is_month_end']     = df_feat.index.is_month_end.astype(int)
    df_feat['is_quarter_start'] = df_feat.index.is_quarter_start.astype(int)
    df_feat['is_quarter_end']   = df_feat.index.is_quarter_end.astype(int)
    if dropna_y:
        df_feat = df_feat.dropna(subset=['y'])
    return df_feat

# ----------------------------------------
# 2. Búsqueda de hiperparámetros y entrenamiento
# ----------------------------------------
param_grid = {
    'learning_rate': [0.01, 0.05],
    'num_leaves':    [31, 63],
    'max_depth':     [7, 15],
    'n_estimators':  [500, 1000]
}

def tune_lgbm(X_tr, y_tr, X_val, y_val):
    best_rmse = float('inf')
    best_params = None
    for lr in param_grid['learning_rate']:
        for nl in param_grid['num_leaves']:
            for md in param_grid['max_depth']:
                for ne in param_grid['n_estimators']:
                    params = {
                        'learning_rate': lr,
                        'num_leaves':    nl,
                        'max_depth':     md,
                        'n_estimators':  ne,
                        'verbose':       -1,
                        'verbosity':     -1,
                        'objective':     'regression'
                    }
                    model = lgb.LGBMRegressor(**params)
                    model.fit(
                        X_tr, y_tr,
                        eval_set=[(X_val, y_val)],
                        eval_metric='l2',
                        callbacks=[
                            lgb.early_stopping(stopping_rounds=50),
                            lgb.log_evaluation(period=0)
                        ]
                    )
                    y_pred_val = model.predict(X_val, num_iteration=model.best_iteration_)
                    rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
                    if rmse_val < best_rmse:
                        best_rmse = rmse_val
                        best_params = params.copy()
    return best_params, best_rmse

@st.cache_data
def train_and_forecast(ts_daily):
    modelos_tuned = {}
    rmse_valid_dict = {}
    for categoria in ts_daily.columns:
        ts_cat = ts_daily[[categoria]].rename(columns={categoria: 'y'})
        df_feat = create_features(ts_cat, dropna_y=True)
        split_date = df_feat.index.max() - timedelta(days=30)
        train_mask = df_feat.index <= split_date
        val_mask   = df_feat.index >  split_date

        X_train = df_feat.loc[train_mask].drop(columns=['y'])
        y_train = df_feat.loc[train_mask, 'y']
        X_val   = df_feat.loc[val_mask].drop(columns=['y'])
        y_val   = df_feat.loc[val_mask, 'y']

        best_params, best_rmse = tune_lgbm(X_train, y_train, X_val, y_val)
        rmse_valid_dict[categoria] = best_rmse

        final_model = lgb.LGBMRegressor(**best_params)
        final_model.fit(
            df_feat.drop(columns=['y']),
            df_feat['y'],
            eval_metric='l2'
        )
        modelos_tuned[categoria] = final_model

    last_date = ts_daily.index.max()
    future_dates = pd.date_range(
        start=last_date + timedelta(days=1),
        periods=30,
        freq='D'
    )
    df_forecast_diario = pd.DataFrame(index=future_dates, columns=ts_daily.columns)

    for categoria, modelo in modelos_tuned.items():
        df_fut = pd.DataFrame(index=future_dates)
        df_fut['y'] = np.nan
        df_fut_feat = create_features(df_fut, dropna_y=False)
        X_future = df_fut_feat.drop(columns=['y'])
        y_pred_future = modelo.predict(X_future, num_iteration=modelo.best_iteration_)
        df_forecast_diario[categoria] = np.round(y_pred_future).astype(int)

    return modelos_tuned, rmse_valid_dict, df_forecast_diario

modelos_tuned, rmse_valid_dict, df_forecast_diario = train_and_forecast(ts_daily)

# ----------------------------------------
# 3. Streamlit Dashboard
# ----------------------------------------
st.title("📊 Dashboard de Efectos de Estacionalidad y Pronóstico de Ventas")

# Sidebar: Filtros Generales
st.sidebar.header("Filtros")

# 3.1. Selector de Categorías
all_categories = list(ts_daily.columns)
selected_categories = st.sidebar.multiselect(
    "Seleccionar Categoría(s):",
    options=all_categories,
    default=all_categories
)

# 3.2. Rango de Fechas Histórico
min_date = ts_daily.index.min().date()
max_date = ts_daily.index.max().date()
date_range = st.sidebar.date_input(
    "Rango de Fechas (Histórico):",
    value=[max_date - timedelta(days=365), max_date],
    min_value=min_date,
    max_value=max_date
)
start_hist, end_hist = map(pd.to_datetime, date_range)

# 3.3. Nivel de Agregación
aggregation = st.sidebar.selectbox(
    "Nivel de Agregación:",
    options=["Diario", "Semanal", "Mensual"],
    index=0
)

# 3.4. Checkbox: Mostrar Pronóstico
show_forecast = st.sidebar.checkbox("Mostrar Pronóstico (próximos 30 días)", value=True)

# 3.5. Carga CSV de Eventos (opcional)
uploaded_file = st.sidebar.file_uploader(
    "Subir CSV de Eventos (fecha, nombre_evento, categoría):",
    type=["csv"]
)
if uploaded_file:
    df_eventos = pd.read_csv(uploaded_file, parse_dates=["fecha"])
else:
    df_eventos = pd.DataFrame(columns=["fecha", "nombre_evento", "categoría"])

st.sidebar.markdown("---")

# ----------------------------------------
# 4. Sección A: Resumen Ejecutivo (KPIs)
# ----------------------------------------
st.header("⭐ Resumen Ejecutivo")

cols_kpi = st.columns(len(selected_categories))
for idx, categoria in enumerate(selected_categories):
    total_actual = ts_daily[categoria].loc[start_hist:end_hist].sum()
    same_period_last_year_start = start_hist - pd.DateOffset(years=1)
    same_period_last_year_end   = end_hist - pd.DateOffset(years=1)
    total_last_year = ts_daily[categoria].loc[
        same_period_last_year_start : same_period_last_year_end
    ].sum()
    pct_change = ((total_actual - total_last_year) / total_last_year * 100) \
        if total_last_year != 0 else np.nan

    if aggregation == "Diario":
        avg_unit = ts_daily[categoria].loc[start_hist:end_hist].mean()
    elif aggregation == "Semanal":
        avg_unit = ts_daily[categoria].loc[start_hist:end_hist].resample("W").sum().mean()
    else:  # Mensual
        avg_unit = ts_daily[categoria].loc[start_hist:end_hist].resample("M").sum().mean()

    with cols_kpi[idx]:
        st.metric(
            label=f"📦 {categoria} - Total Ventas",
            value=f"{total_actual:,}",
            delta=f"{pct_change:.2f}%" if not np.isnan(pct_change) else "N/A"
        )
        st.write(f"Promedio ({aggregation}): {avg_unit:.0f}")

st.markdown("---")

# ----------------------------------------
# 5. Sección B: Serie Histórica Multi-Categoría
# ----------------------------------------
st.header("📈 Serie Histórica de Ventas")

df_plot = ts_daily[selected_categories].loc[start_hist:end_hist]
if aggregation == "Semanal":
    df_plot = df_plot.resample("W").sum()
elif aggregation == "Mensual":
    df_plot = df_plot.resample("M").sum()

# Si no hay eventos, usamos simplemente line_chart de Streamlit
if df_eventos.empty:
    st.line_chart(df_plot)
else:
    # Con eventos, pasamos a Plotly para superponer líneas verticales
    categorias_evt = st.selectbox(
        "Seleccionar evento para resaltar:",
        options=["Todos"] + df_eventos["nombre_evento"].unique().tolist()
    )
    fig_hist = go.Figure()
    for cat in selected_categories:
        serie_cat = df_plot[cat]
        fig_hist.add_trace(go.Scatter(
            x=serie_cat.index,
            y=serie_cat.values,
            mode='lines',
            name=cat
        ))
    # Dibujamos cada evento como línea vertical
    for _, row in df_eventos.iterrows():
        if categorias_evt == "Todos" or row["nombre_evento"] == categorias_evt:
            if start_hist <= row["fecha"] <= end_hist:
                fig_hist.add_vline(
                    x=row["fecha"],
                    line=dict(color="red", width=1, dash="dot"),
                    annotation_text=row["nombre_evento"],
                    annotation_position="top left"
                )
    st.plotly_chart(fig_hist, use_container_width=True)

st.markdown("---")

# ----------------------------------------
# 6. Sección C: Descomposición de Estacionalidad
# ----------------------------------------
st.header("🔍 Descomposición de Estacionalidad")

decomp_cat = st.selectbox(
    "Seleccionar Categoría para Descomposición:",
    options=selected_categories
)
years_to_display = st.slider(
    "Últimos N años para descomposición:",
    min_value=1, max_value=5, value=2, step=1
)
period_map = {"Diario": 365, "Semanal": 52, "Mensual": 12}

# Selección de la serie recortada
serie = ts_daily[decomp_cat].loc[
    ts_daily.index.max() - pd.DateOffset(years=years_to_display) : ts_daily.index.max()
]
if aggregation == "Semanal":
    serie = serie.resample("W").sum()
elif aggregation == "Mensual":
    serie = serie.resample("M").sum()

if len(serie) >= period_map[aggregation] * 2:
    result = seasonal_decompose(serie, model="additive", period=period_map[aggregation])

    # Construimos un DataFrame con las 3 series: trend, seasonal, resid
    df_decomp = pd.DataFrame({
        "Tendencia":    result.trend,
        "Estacionalidad": result.seasonal,
        "Residuo":      result.resid
    })

    st.subheader("Componentes de la Descomposición")
    st.write("""
    - **Tendencia:** Variación lenta a lo largo del tiempo.  
    - **Estacionalidad:** Patrón repetitivo en períodos iguales.  
    - **Residuo:** Lo que no se explica por tendencia ni estacionalidad.  
    """)
    st.line_chart(df_decomp.fillna(method="bfill"))  # interpolar valores nulos para mostrar líneas
else:
    st.write("Serie demasiado corta para descomposición con este nivel de agregación.")

st.markdown("---")

# ----------------------------------------
# 7. Sección D: Impacto de Eventos Especiales
# ----------------------------------------
st.header("📊 Impacto de Eventos Especiales")

if df_eventos.empty:
    st.info("No se cargaron eventos. Por favor sube un CSV de eventos en la barra lateral.")
else:
    eventos_selec = st.multiselect(
        "Seleccionar Evento(s) para Análisis:",
        options=df_eventos["nombre_evento"].unique().tolist()
    )
    window_days = st.slider(
        "Ventana de días previos y posteriores:",
        min_value=7, max_value=30, value=14, step=7
    )
    if eventos_selec:
        impacto_list = []
        for evento in eventos_selec:
            row_evt = df_eventos[df_eventos["nombre_evento"] == evento].iloc[0]
            fecha_evt = row_evt["fecha"]
            cat_evt = row_evt["categoría"]
            if cat_evt not in selected_categories and cat_evt != "Todas":
                continue

            ventas_pre = ts_daily[cat_evt].loc[
                fecha_evt - pd.Timedelta(days=window_days) : fecha_evt - pd.Timedelta(days=1)
            ].mean()
            ventas_post = ts_daily[cat_evt].loc[
                fecha_evt + pd.Timedelta(days=1) : fecha_evt + pd.Timedelta(days=window_days)
            ].mean()
            ventas_evt = ts_daily[cat_evt].loc[fecha_evt]

            delta_pct = ((ventas_evt - ventas_pre) / ventas_pre * 100) if ventas_pre != 0 else np.nan
            impacto_list.append({
                "Evento":           evento,
                "Categoría":        cat_evt,
                "Ventas Previo":    ventas_pre,
                "Ventas Evento":    ventas_evt,
                "Ventas Posterior": ventas_post,
                "Δ (%)":            delta_pct
            })
        df_impacto = pd.DataFrame(impacto_list)
        if not df_impacto.empty:
            st.subheader("Tabla de Impacto de Eventos")
            st.dataframe(df_impacto.round(2))
            st.subheader("Variación porcentual en Ventas (%)")
            st.bar_chart(df_impacto.set_index("Evento")["Δ (%)"])
        else:
            st.write("No hay eventos aplicables a las categorías seleccionadas.")

st.markdown("---")

# ----------------------------------------
# 8. Sección E: Pronóstico vs Real Histórico
# ----------------------------------------
st.header("🔮 Pronóstico vs Real Histórico")

cat_forecast = st.selectbox(
    "Seleccionar Categoría para Pronóstico:",
    options=selected_categories
)
last_n_days = st.slider(
    "Mostrar últimos N días de histórico:",
    min_value=30, max_value=90, value=60, step=10
)

# 8.1. Serie histórica recortada
historic_series = ts_daily[cat_forecast].dropna()[-last_n_days:]

# 8.2. Serie de pronóstico (30 días)
if show_forecast:
    forecast_series = df_forecast_diario[cat_forecast]
else:
    forecast_series = pd.Series(dtype=float)  # vacío si no se quiere mostrar pronóstico

# Construimos gráfico con Plotly para incluir línea vertical y etiquetas
fig_fc = go.Figure()

# Trazamos histórico
fig_fc.add_trace(go.Scatter(
    x=historic_series.index,
    y=historic_series.values,
    mode='lines',
    name="Histórico",
    line=dict(color='blue')
))

# Trazamos pronóstico (solo si está activo)
if show_forecast:
    fig_fc.add_trace(go.Scatter(
        x=forecast_series.index,
        y=forecast_series.values,
        mode='lines',
        name="Pronóstico",
        line=dict(color='orange', dash='dash')
    ))

    # 1) Dibujamos la línea vertical “Hoy” con add_shape
    last_date = ts_daily.index.max()
    fig_fc.add_shape(
        type="line",
        x0=last_date,
        x1=last_date,
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        line=dict(color="gray", width=1, dash="dot")
    )
    # 2) Agregamos la etiqueta con add_annotation
    fig_fc.add_annotation(
        x=last_date,
        y=1.0,
        xref="x",
        yref="paper",
        text="Hoy",
        showarrow=False,
        yshift=10
    )

fig_fc.update_layout(
    title=f"Histórico vs Pronóstico - {cat_forecast}",
    xaxis_title="Fecha",
    yaxis_title="Ventas diarias",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig_fc, use_container_width=True)
