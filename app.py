
import re
from datetime import datetime, date
from io import StringIO
from zoneinfo import ZoneInfo

import holidays
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
from pandas.tseries.offsets import CustomBusinessDay
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

st.set_page_config(
    page_title="Proyección TRM Colombia",
    page_icon="💵",
    layout="wide"
)

FUENTE_PRINCIPAL_TRM = "Datos Abiertos Colombia - TRM histórica mcec-87by"
URL_TRM = "https://www.datos.gov.co/resource/mcec-87by.json?$limit=50000"
URL_DOLAR_COLOMBIA = "https://www.dolar-colombia.com/"
URL_DOLAR_GLOBAL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTWEXBGS"
URL_PETROLEO_WTI = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DCOILWTICO"


# ============================================================
# FUNCIONES DE FORMATO
# ============================================================

def formato_fecha(fecha):
    return pd.to_datetime(fecha).strftime("%d/%m/%Y")


def formato_cop(valor):
    if valor is None or pd.isna(valor):
        return "NO DISPONIBLE"
    if valor < 0:
        return "-$" + f"{abs(valor):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".") + " COP"
    return "$" + f"{valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".") + " COP"


def formato_porcentaje(valor):
    if valor is None or pd.isna(valor):
        return "NO DISPONIBLE"
    return f"{valor:,.2f}%".replace(",", "X").replace(".", ",").replace("X", ".")


def convertir_numero_cop(texto):
    """
    Convierte textos como:
    3,440.83
    3.440,83
    3440.83
    3440,83
    a float.
    """
    s = str(texto).strip()
    s = s.replace(" ", "")
    s = re.sub(r"[^0-9\.,]", "", s)

    if s == "":
        return None

    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        s = s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(",", "")

    try:
        return float(s)
    except Exception:
        return None


# ============================================================
# CALENDARIO COLOMBIA
# ============================================================

def es_fecha_habil_colombia(fecha):
    fecha = pd.to_datetime(fecha).normalize()
    festivos_co = holidays.country_holidays(
        "CO",
        years=[fecha.year, fecha.year + 1],
        language="es"
    )
    return fecha.weekday() < 5 and fecha.date() not in festivos_co


def ajustar_a_siguiente_habil(fecha):
    fecha = pd.to_datetime(fecha).normalize()
    while not es_fecha_habil_colombia(fecha):
        fecha = fecha + pd.Timedelta(days=1)
    return fecha


def proxima_fecha_habil_colombia(fecha):
    fecha = pd.to_datetime(fecha).normalize()
    anios = [fecha.year, fecha.year + 1]
    festivos_co = holidays.country_holidays("CO", years=anios, language="es")
    fechas_festivas = pd.to_datetime(list(festivos_co.keys()))

    calendario_colombia = CustomBusinessDay(
        weekmask="Mon Tue Wed Thu Fri",
        holidays=fechas_festivas
    )

    return fecha + calendario_colombia


# ============================================================
# DESCARGA Y VALIDACIÓN DE DATOS
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def descargar_trm():
    df_raw_trm = pd.read_json(URL_TRM)

    df_trm = df_raw_trm.copy()
    df_trm = df_trm.rename(columns={
        "valor": "trm",
        "vigenciadesde": "fecha_desde",
        "vigenciahasta": "fecha_hasta"
    })

    df_trm["fecha_desde"] = pd.to_datetime(df_trm["fecha_desde"], errors="coerce").dt.normalize()
    df_trm["fecha_hasta"] = pd.to_datetime(df_trm["fecha_hasta"], errors="coerce").dt.normalize()
    df_trm["trm"] = pd.to_numeric(df_trm["trm"], errors="coerce")

    df_trm = df_trm.dropna(subset=["trm", "fecha_desde", "fecha_hasta"])
    df_trm = df_trm.sort_values("fecha_desde").copy()
    df_trm = df_trm.drop_duplicates(subset=["fecha_desde", "fecha_hasta"], keep="last")

    return df_trm


@st.cache_data(ttl=3600, show_spinner=False)
def descargar_variables_externas():
    df_dolar_global = pd.read_csv(URL_DOLAR_GLOBAL)
    df_petroleo = pd.read_csv(URL_PETROLEO_WTI)

    df_dolar_global = df_dolar_global.rename(columns={
        "observation_date": "fecha",
        "DTWEXBGS": "indice_dolar_global"
    })

    df_petroleo = df_petroleo.rename(columns={
        "observation_date": "fecha",
        "DCOILWTICO": "petroleo_wti"
    })

    df_dolar_global["fecha"] = pd.to_datetime(df_dolar_global["fecha"], errors="coerce").dt.normalize()
    df_dolar_global["indice_dolar_global"] = pd.to_numeric(df_dolar_global["indice_dolar_global"], errors="coerce")
    df_dolar_global = df_dolar_global.dropna().sort_values("fecha")

    df_petroleo["fecha"] = pd.to_datetime(df_petroleo["fecha"], errors="coerce").dt.normalize()
    df_petroleo["petroleo_wti"] = pd.to_numeric(df_petroleo["petroleo_wti"], errors="coerce")
    df_petroleo = df_petroleo.dropna().sort_values("fecha")

    return df_dolar_global, df_petroleo


@st.cache_data(ttl=1800, show_spinner=False)
def obtener_validacion_dolar_colombia(valor_oficial):
    """
    Fuente auxiliar gratuita.
    No se usa como fuente oficial principal.
    Sirve únicamente para contraste del valor visible en la página.
    """
    resultado = {
        "fuente_auxiliar": URL_DOLAR_COLOMBIA,
        "valor_auxiliar": None,
        "estado_auxiliar": "NO DISPONIBLE",
        "diferencia_auxiliar": None,
        "mensaje_auxiliar": "No fue posible leer la fuente auxiliar."
    }

    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(URL_DOLAR_COLOMBIA, headers=headers, timeout=20)
        html = resp.text

        patrones = [
            r"1\s*USD\s*=\s*(?:[↓↑]\s*)?([0-9][0-9\.,]*)\s*COP",
            r"1\s*USD\s*=\s*(?:[↓↑]\s*)?([0-9][0-9\.,]*)"
        ]

        valor_aux = None

        for patron in patrones:
            m = re.search(patron, html, flags=re.IGNORECASE)
            if m:
                valor_aux = convertir_numero_cop(m.group(1))
                break

        if valor_aux is None:
            texto = re.sub(r"<[^>]+>", " ", html)
            texto = re.sub(r"\s+", " ", texto)

            for patron in patrones:
                m = re.search(patron, texto, flags=re.IGNORECASE)
                if m:
                    valor_aux = convertir_numero_cop(m.group(1))
                    break

        if valor_aux is None:
            return resultado

        diferencia = valor_aux - valor_oficial

        resultado["valor_auxiliar"] = valor_aux
        resultado["diferencia_auxiliar"] = diferencia

        if abs(diferencia) <= 0.05:
            resultado["estado_auxiliar"] = "COINCIDE"
            resultado["mensaje_auxiliar"] = "La fuente auxiliar coincide con la fuente principal."
        else:
            resultado["estado_auxiliar"] = "NO COINCIDE"
            resultado["mensaje_auxiliar"] = (
                "La fuente auxiliar no coincide con la fuente principal. "
                "Se conserva la fuente principal oficial/base pública."
            )

        return resultado

    except Exception as e:
        resultado["mensaje_auxiliar"] = f"No fue posible validar con la fuente auxiliar. Detalle: {e}"
        return resultado


# ============================================================
# MODELO
# ============================================================

def crear_variables_modelo(base):
    base = base.sort_values("fecha").copy()

    base["variacion_trm"] = base["trm"].diff()
    base["variacion_pct"] = base["trm"].pct_change() * 100

    base["media_3"] = base["trm"].rolling(3).mean()
    base["media_5"] = base["trm"].rolling(5).mean()
    base["media_10"] = base["trm"].rolling(10).mean()
    base["media_20"] = base["trm"].rolling(20).mean()
    base["media_60"] = base["trm"].rolling(60).mean()

    base["volatilidad_5"] = base["variacion_trm"].rolling(5).std()
    base["volatilidad_20"] = base["variacion_trm"].rolling(20).std()

    base["trm_lag_1"] = base["trm"].shift(1)
    base["trm_lag_2"] = base["trm"].shift(2)
    base["trm_lag_3"] = base["trm"].shift(3)
    base["trm_lag_5"] = base["trm"].shift(5)
    base["trm_lag_10"] = base["trm"].shift(10)
    base["trm_lag_20"] = base["trm"].shift(20)

    base["cambio_5"] = base["trm"] - base["trm_lag_5"]
    base["cambio_20"] = base["trm"] - base["trm_lag_20"]

    base["dia_semana"] = base["fecha"].dt.weekday
    base["mes"] = base["fecha"].dt.month

    base["dias_sin_actualizar_indice_dolar"] = (
        base["fecha"] - base["fecha_indice_dolar"]
    ).dt.days

    base["dias_sin_actualizar_petroleo"] = (
        base["fecha"] - base["fecha_petroleo"]
    ).dt.days

    base["var_indice_dolar"] = base["indice_dolar_global"].pct_change() * 100
    base["var_petroleo_wti"] = base["petroleo_wti"].pct_change() * 100

    return base


def preparar_base_modelo(df_trm_permitida, df_dolar_global, df_petroleo):
    df_pub = df_trm_permitida.copy()
    df_pub["fecha"] = df_pub["fecha_desde"]
    df_pub = df_pub.sort_values("fecha").copy()

    df_pub["variacion_trm"] = df_pub["trm"].diff()
    df_pub["variacion_pct"] = df_pub["trm"].pct_change() * 100

    df_pub["media_3"] = df_pub["trm"].rolling(3).mean()
    df_pub["media_5"] = df_pub["trm"].rolling(5).mean()
    df_pub["media_10"] = df_pub["trm"].rolling(10).mean()
    df_pub["media_20"] = df_pub["trm"].rolling(20).mean()
    df_pub["media_60"] = df_pub["trm"].rolling(60).mean()

    df_pub["volatilidad_5"] = df_pub["variacion_trm"].rolling(5).std()
    df_pub["volatilidad_20"] = df_pub["variacion_trm"].rolling(20).std()

    df_pub["trm_lag_1"] = df_pub["trm"].shift(1)
    df_pub["trm_lag_2"] = df_pub["trm"].shift(2)
    df_pub["trm_lag_3"] = df_pub["trm"].shift(3)
    df_pub["trm_lag_5"] = df_pub["trm"].shift(5)
    df_pub["trm_lag_10"] = df_pub["trm"].shift(10)
    df_pub["trm_lag_20"] = df_pub["trm"].shift(20)

    df_pub["cambio_5"] = df_pub["trm"] - df_pub["trm_lag_5"]
    df_pub["cambio_20"] = df_pub["trm"] - df_pub["trm_lag_20"]

    df_pub["dia_semana"] = df_pub["fecha"].dt.weekday
    df_pub["mes"] = df_pub["fecha"].dt.month

    df_dolar_aux = df_dolar_global.rename(columns={"fecha": "fecha_indice_dolar"}).sort_values("fecha_indice_dolar")
    df_petroleo_aux = df_petroleo.rename(columns={"fecha": "fecha_petroleo"}).sort_values("fecha_petroleo")

    df_modelo = pd.merge_asof(
        df_pub.sort_values("fecha"),
        df_dolar_aux,
        left_on="fecha",
        right_on="fecha_indice_dolar",
        direction="backward"
    )

    df_modelo = pd.merge_asof(
        df_modelo.sort_values("fecha"),
        df_petroleo_aux,
        left_on="fecha",
        right_on="fecha_petroleo",
        direction="backward"
    )

    df_modelo["dias_sin_actualizar_indice_dolar"] = (
        df_modelo["fecha"] - df_modelo["fecha_indice_dolar"]
    ).dt.days

    df_modelo["dias_sin_actualizar_petroleo"] = (
        df_modelo["fecha"] - df_modelo["fecha_petroleo"]
    ).dt.days

    df_modelo["var_indice_dolar"] = df_modelo["indice_dolar_global"].pct_change() * 100
    df_modelo["var_petroleo_wti"] = df_modelo["petroleo_wti"].pct_change() * 100

    df_modelo["trm_siguiente"] = df_modelo["trm"].shift(-1)
    df_modelo["cambio_siguiente"] = df_modelo["trm_siguiente"] - df_modelo["trm"]

    columnas_modelo = [
        "trm",
        "variacion_trm",
        "variacion_pct",
        "media_3",
        "media_5",
        "media_10",
        "media_20",
        "media_60",
        "volatilidad_5",
        "volatilidad_20",
        "trm_lag_1",
        "trm_lag_2",
        "trm_lag_3",
        "trm_lag_5",
        "trm_lag_10",
        "trm_lag_20",
        "cambio_5",
        "cambio_20",
        "dia_semana",
        "mes",
        "indice_dolar_global",
        "petroleo_wti",
        "dias_sin_actualizar_indice_dolar",
        "dias_sin_actualizar_petroleo",
        "var_indice_dolar",
        "var_petroleo_wti"
    ]

    return df_modelo, columnas_modelo


@st.cache_resource(show_spinner=False)
def entrenar_modelo(df_modelo_csv, columnas_modelo):
    df_modelo = pd.read_json(StringIO(df_modelo_csv), orient="split")
    for col in ["fecha", "fecha_indice_dolar", "fecha_petroleo", "fecha_desde", "fecha_hasta"]:
        if col in df_modelo.columns:
            df_modelo[col] = pd.to_datetime(df_modelo[col], errors="coerce")

    df_entrenamiento = df_modelo.dropna(
        subset=columnas_modelo + ["cambio_siguiente", "trm_siguiente"]
    ).copy()

    if len(df_entrenamiento) < 500:
        raise Exception("No hay suficientes datos para entrenar el modelo.")

    X = df_entrenamiento[columnas_modelo]
    y_cambio = df_entrenamiento["cambio_siguiente"]
    y_trm_real = df_entrenamiento["trm_siguiente"]

    tamanio_prueba = min(365, int(len(df_entrenamiento) * 0.2))

    X_train = X.iloc[:-tamanio_prueba]
    X_test = X.iloc[-tamanio_prueba:]

    y_train = y_cambio.iloc[:-tamanio_prueba]
    y_test_trm = y_trm_real.iloc[-tamanio_prueba:]

    trm_actual_test = df_entrenamiento["trm"].iloc[-tamanio_prueba:]

    modelos = {
        "Random Forest": RandomForestRegressor(
            n_estimators=400,
            random_state=42,
            min_samples_leaf=3,
            n_jobs=-1
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=400,
            random_state=42,
            min_samples_leaf=3,
            n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42)
    }

    resultados = []

    for nombre, modelo in modelos.items():
        modelo.fit(X_train, y_train)

        pred_cambio = modelo.predict(X_test)
        pred_trm = trm_actual_test.values + pred_cambio

        mae = mean_absolute_error(y_test_trm, pred_trm)
        rmse = np.sqrt(mean_squared_error(y_test_trm, pred_trm))

        resultados.append({
            "modelo": nombre,
            "mae": mae,
            "rmse": rmse,
            "objeto": modelo,
            "registros_entrenamiento": len(df_entrenamiento)
        })

    mejor_modelo = min(resultados, key=lambda x: x["mae"])
    return mejor_modelo


def proyectar_fecha(
    df_modelo,
    columnas_modelo,
    mejor_modelo,
    fecha_hoy_colombia,
    fecha_objetivo,
    fecha_base_modelo,
    trm_base_modelo
):
    hist = df_modelo[[
        "fecha",
        "trm",
        "indice_dolar_global",
        "petroleo_wti",
        "fecha_indice_dolar",
        "fecha_petroleo"
    ]].dropna().copy()

    hist = hist[hist["fecha"] <= fecha_base_modelo].sort_values("fecha").copy()

    proyecciones = []

    fecha_actual = fecha_base_modelo
    contador = 0

    while fecha_actual < fecha_objetivo:
        contador += 1

        if contador > 520:
            raise Exception(
                "La fecha objetivo está demasiado lejana para esta versión del modelo. "
                "Usa una fecha menor a aproximadamente dos años hábiles."
            )

        aux = crear_variables_modelo(hist)
        fila_actual = aux.dropna(subset=columnas_modelo).sort_values("fecha").iloc[-1]

        X_actual = fila_actual[columnas_modelo].to_frame().T

        cambio_predicho = mejor_modelo["objeto"].predict(X_actual)[0]
        trm_predicha = fila_actual["trm"] + cambio_predicho

        fecha_siguiente = proxima_fecha_habil_colombia(fila_actual["fecha"])

        proyecciones.append({
            "fecha": fecha_siguiente,
            "fecha_formato": formato_fecha(fecha_siguiente),
            "trm_estimada": trm_predicha,
            "cambio_diario_estimado": cambio_predicho
        })

        nueva_fila = {
            "fecha": fecha_siguiente,
            "trm": trm_predicha,
            "indice_dolar_global": fila_actual["indice_dolar_global"],
            "petroleo_wti": fila_actual["petroleo_wti"],
            "fecha_indice_dolar": fila_actual["fecha_indice_dolar"],
            "fecha_petroleo": fila_actual["fecha_petroleo"]
        }

        hist = pd.concat([hist, pd.DataFrame([nueva_fila])], ignore_index=True)
        fecha_actual = fecha_siguiente

    return pd.DataFrame(proyecciones)


# ============================================================
# INTERFAZ
# ============================================================

st.title("💵 Proyección TRM Colombia")
st.caption("TRM oficial + validación auxiliar + proyección estadística por fecha")

with st.sidebar:
    st.header("Parámetros")
    fecha_default = date.today()
    fecha_usuario = st.date_input(
        "Fecha objetivo",
        value=fecha_default,
        format="DD/MM/YYYY"
    )

    ejecutar = st.button("Generar análisis", type="primary")

    st.divider()
    st.caption("Formato de fecha: DD/MM/AAAA")
    st.caption("La TRM oficial es diaria, no intradía.")
    st.caption("Dólar-Colombia se usa solo como fuente auxiliar.")

if not ejecutar:
    st.info("Selecciona la fecha objetivo en la barra lateral y pulsa **Generar análisis**.")
    st.stop()

fecha_solicitada = pd.Timestamp(fecha_usuario).normalize()

with st.spinner("Consultando fuentes y preparando modelo..."):
    fecha_hoy_colombia = pd.Timestamp(datetime.now(ZoneInfo("America/Bogota")).date())
    fecha_objetivo = ajustar_a_siguiente_habil(fecha_solicitada)

    df_trm = descargar_trm()
    df_futuros_bloqueados = df_trm[df_trm["fecha_desde"] > fecha_hoy_colombia].copy()
    df_trm_permitida = df_trm[df_trm["fecha_desde"] <= fecha_hoy_colombia].copy()

    if len(df_trm_permitida) == 0:
        st.error("No hay TRM vigente o pasada disponible para la fecha de consulta.")
        st.stop()

    df_vigente_hoy = df_trm_permitida[
        (df_trm_permitida["fecha_desde"] <= fecha_hoy_colombia) &
        (df_trm_permitida["fecha_hasta"] >= fecha_hoy_colombia)
    ].copy()

    if len(df_vigente_hoy) > 0:
        fila_vigente = df_vigente_hoy.iloc[-1]
    else:
        fila_vigente = df_trm_permitida.iloc[-1]

    trm_vigente_hoy = fila_vigente["trm"]
    vigente_desde = fila_vigente["fecha_desde"]
    vigente_hasta = fila_vigente["fecha_hasta"]

    validacion_aux = obtener_validacion_dolar_colombia(trm_vigente_hoy)

    df_dolar_global, df_petroleo = descargar_variables_externas()
    df_modelo, columnas_modelo = preparar_base_modelo(df_trm_permitida, df_dolar_global, df_petroleo)

    df_modelo_csv = df_modelo.to_json(orient="split", date_format="iso")
    mejor_modelo = entrenar_modelo(df_modelo_csv, columnas_modelo)

    df_pred = df_modelo.dropna(subset=columnas_modelo).copy()
    df_pred = df_pred[df_pred["fecha"] <= fecha_hoy_colombia].sort_values("fecha")

    ultima_fila = df_pred.iloc[-1]
    fecha_base_modelo = ultima_fila["fecha"]
    trm_base_modelo = ultima_fila["trm"]

# ============================================================
# BLOQUE TRM OFICIAL
# ============================================================

st.subheader("TRM oficial vigente")

col1, col2, col3 = st.columns(3)

col1.metric("TRM vigente", formato_cop(trm_vigente_hoy))
col2.metric("Vigente desde", formato_fecha(vigente_desde))
col3.metric("Vigente hasta", formato_fecha(vigente_hasta))

st.write(f"**Fuente principal:** {FUENTE_PRINCIPAL_TRM}")
st.write("**Uso de registros futuros:** NO")
st.write("**Valor futuro mostrado:** NO")

if len(df_futuros_bloqueados) > 0:
    st.warning(f"Registros futuros detectados y bloqueados: {len(df_futuros_bloqueados)}")
else:
    st.success("No se detectaron registros futuros para bloquear.")

# ============================================================
# VALIDACIÓN AUXILIAR
# ============================================================

st.subheader("Validación auxiliar")

col_a, col_b, col_c = st.columns(3)

col_a.metric(
    "Dólar-Colombia",
    formato_cop(validacion_aux["valor_auxiliar"]) if validacion_aux["valor_auxiliar"] is not None else "NO DISPONIBLE"
)
col_b.metric(
    "Diferencia",
    formato_cop(validacion_aux["diferencia_auxiliar"]) if validacion_aux["diferencia_auxiliar"] is not None else "NO DISPONIBLE"
)
col_c.metric("Estado", validacion_aux["estado_auxiliar"])

if validacion_aux["estado_auxiliar"] == "COINCIDE":
    st.success(validacion_aux["mensaje_auxiliar"])
elif validacion_aux["estado_auxiliar"] == "NO COINCIDE":
    st.warning(validacion_aux["mensaje_auxiliar"])
else:
    st.info(validacion_aux["mensaje_auxiliar"])

st.caption("La fuente auxiliar no reemplaza la fuente principal; solo ayuda a contrastar el valor visible en una fuente gratuita.")

# ============================================================
# CONSULTA HISTÓRICA O PROYECCIÓN
# ============================================================

if fecha_solicitada <= fecha_base_modelo:
    st.subheader("Consulta histórica")

    df_busqueda = df_trm[
        (df_trm["fecha_desde"] <= fecha_solicitada) &
        (df_trm["fecha_hasta"] >= fecha_solicitada)
    ].copy()

    if len(df_busqueda) == 0:
        st.error("No se encontró TRM oficial histórica para la fecha solicitada.")
        st.stop()

    fila_hist = df_busqueda.iloc[-1]

    st.metric("TRM oficial encontrada", formato_cop(fila_hist["trm"]))
    st.write(f"**Fecha solicitada:** {formato_fecha(fecha_solicitada)}")
    st.write(f"**Vigente desde:** {formato_fecha(fila_hist['fecha_desde'])}")
    st.write(f"**Vigente hasta:** {formato_fecha(fila_hist['fecha_hasta'])}")

    df_hist_salida = pd.DataFrame([{
        "fecha_solicitada": formato_fecha(fecha_solicitada),
        "trm_oficial": fila_hist["trm"],
        "vigente_desde": formato_fecha(fila_hist["fecha_desde"]),
        "vigente_hasta": formato_fecha(fila_hist["fecha_hasta"]),
        "fuente_principal": FUENTE_PRINCIPAL_TRM,
        "valor_auxiliar_dolar_colombia": validacion_aux["valor_auxiliar"],
        "estado_validacion_auxiliar": validacion_aux["estado_auxiliar"],
        "registros_futuros_bloqueados": len(df_futuros_bloqueados)
    }])

    st.dataframe(df_hist_salida, use_container_width=True)

    csv_hist = df_hist_salida.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Descargar consulta histórica CSV",
        data=csv_hist,
        file_name=f"consulta_historica_trm_{fecha_solicitada.strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

else:
    st.subheader("Proyección solicitada")

    with st.spinner("Proyectando día hábil por día hábil..."):
        df_proyeccion = proyectar_fecha(
            df_modelo=df_modelo,
            columnas_modelo=columnas_modelo,
            mejor_modelo=mejor_modelo,
            fecha_hoy_colombia=fecha_hoy_colombia,
            fecha_objetivo=fecha_objetivo,
            fecha_base_modelo=fecha_base_modelo,
            trm_base_modelo=trm_base_modelo
        )

    fila_final = df_proyeccion[df_proyeccion["fecha"] <= fecha_objetivo].iloc[-1]

    trm_estimacion_final = fila_final["trm_estimada"]
    cambio_total = trm_estimacion_final - trm_base_modelo
    cambio_porcentual = (cambio_total / trm_base_modelo) * 100

    dias_habiles = len(df_proyeccion)
    dias_calendario = (fecha_objetivo - fecha_base_modelo).days
    cambio_promedio_habil = cambio_total / dias_habiles

    factor_horizonte = np.sqrt(dias_habiles)

    error_80 = mejor_modelo["mae"] * 1.5 * factor_horizonte
    error_90 = mejor_modelo["mae"] * 2.0 * factor_horizonte

    rango_bajo_80 = trm_estimacion_final - error_80
    rango_alto_80 = trm_estimacion_final + error_80

    rango_bajo_90 = trm_estimacion_final - error_90
    rango_alto_90 = trm_estimacion_final + error_90

    if cambio_total > 20:
        senal = "ALCISTA"
        lectura = "El modelo estima una subida relevante frente a la TRM actual."
    elif cambio_total < -20:
        senal = "BAJISTA"
        lectura = "El modelo estima una bajada relevante frente a la TRM actual."
    else:
        senal = "LATERAL / CAMBIO MODERADO"
        lectura = "El modelo estima un movimiento moderado, sin una tendencia fuerte."

    if dias_habiles <= 5:
        nivel_confianza = "Mayor confianza relativa, porque el horizonte es corto."
    elif dias_habiles <= 20:
        nivel_confianza = "Confianza media, porque el horizonte ya depende de más supuestos."
    else:
        nivel_confianza = "Confianza baja, porque la fecha está lejana y pueden cambiar petróleo, dólar global, tasas y noticias."

    colx1, colx2, colx3 = st.columns(3)
    colx1.metric("Fecha objetivo", formato_fecha(fecha_objetivo))
    colx2.metric("TRM estimada central", formato_cop(trm_estimacion_final))
    colx3.metric("Cambio estimado", formato_cop(cambio_total), formato_porcentaje(cambio_porcentual))

    st.write(f"**Fecha base del modelo:** {formato_fecha(fecha_base_modelo)}")
    st.write(f"**TRM base del modelo:** {formato_cop(trm_base_modelo)}")
    st.write(f"**Días calendario:** {dias_calendario}")
    st.write(f"**Días hábiles proyectados:** {dias_habiles}")
    st.write(f"**Cambio promedio por día hábil:** {formato_cop(cambio_promedio_habil)}")
    st.write(f"**Señal:** {senal}")
    st.write(f"**Lectura:** {lectura}")
    st.write(f"**Nivel de confianza:** {nivel_confianza}")

    st.subheader("Escenarios")

    df_escenarios = pd.DataFrame({
        "Escenario": [
            "Bajo 90%",
            "Bajo 80%",
            "Central estimado",
            "Alto 80%",
            "Alto 90%"
        ],
        "TRM estimada": [
            rango_bajo_90,
            rango_bajo_80,
            trm_estimacion_final,
            rango_alto_80,
            rango_alto_90
        ]
    })

    df_escenarios["Valor formateado"] = df_escenarios["TRM estimada"].apply(formato_cop)
    st.dataframe(df_escenarios[["Escenario", "Valor formateado"]], use_container_width=True)

    df_decision = pd.DataFrame({
        "Concepto": [
            "Fecha de consulta Colombia",
            "TRM oficial vigente hoy",
            "Vigente desde",
            "Vigente hasta",
            "Fuente principal",
            "Fuente auxiliar",
            "Valor auxiliar leído",
            "Estado validación auxiliar",
            "Registros futuros detectados",
            "Uso de registros futuros",
            "Valor futuro mostrado",
            "Fecha base del modelo",
            "TRM base del modelo",
            "Fecha solicitada",
            "Fecha objetivo usada",
            "Días calendario",
            "Días hábiles proyectados",
            "TRM estimada central",
            "Cambio estimado en pesos",
            "Cambio porcentual",
            "Cambio promedio por día hábil",
            "Escenario bajo 80%",
            "Escenario alto 80%",
            "Escenario bajo 90%",
            "Escenario alto 90%",
            "Señal",
            "Nivel de confianza",
            "Mejor modelo",
            "Error promedio histórico"
        ],
        "Resultado": [
            formato_fecha(fecha_hoy_colombia),
            formato_cop(trm_vigente_hoy),
            formato_fecha(vigente_desde),
            formato_fecha(vigente_hasta),
            FUENTE_PRINCIPAL_TRM,
            validacion_aux["fuente_auxiliar"],
            formato_cop(validacion_aux["valor_auxiliar"]) if validacion_aux["valor_auxiliar"] is not None else "NO DISPONIBLE",
            validacion_aux["estado_auxiliar"],
            str(len(df_futuros_bloqueados)),
            "NO",
            "NO",
            formato_fecha(fecha_base_modelo),
            formato_cop(trm_base_modelo),
            formato_fecha(fecha_solicitada),
            formato_fecha(fecha_objetivo),
            str(dias_calendario),
            str(dias_habiles),
            formato_cop(trm_estimacion_final),
            formato_cop(cambio_total),
            formato_porcentaje(cambio_porcentual),
            formato_cop(cambio_promedio_habil),
            formato_cop(rango_bajo_80),
            formato_cop(rango_alto_80),
            formato_cop(rango_bajo_90),
            formato_cop(rango_alto_90),
            senal,
            nivel_confianza,
            mejor_modelo["modelo"],
            formato_cop(mejor_modelo["mae"])
        ]
    })

    st.subheader("Tabla de decisión")
    st.dataframe(df_decision, use_container_width=True)

    st.subheader("Gráfica")

    df_grafica = df_proyeccion.copy().sort_values("fecha")
    df_base = pd.DataFrame([{
        "fecha": fecha_base_modelo,
        "trm_estimada": trm_base_modelo
    }])

    df_grafica_total = pd.concat([
        df_base,
        df_grafica[["fecha", "trm_estimada"]]
    ], ignore_index=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        df_grafica_total["fecha"],
        df_grafica_total["trm_estimada"],
        marker="o",
        label="Proyección central"
    )

    ax.axhline(rango_bajo_80, linestyle="--", label="Escenario bajo 80%")
    ax.axhline(rango_alto_80, linestyle="--", label="Escenario alto 80%")
    ax.axhline(rango_bajo_90, linestyle=":", label="Escenario bajo 90%")
    ax.axhline(rango_alto_90, linestyle=":", label="Escenario alto 90%")

    ax.set_title("Proyección automática de TRM a fecha específica")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("COP/USD")
    ax.grid(True)
    ax.legend()
    plt.xticks(rotation=45)

    st.pyplot(fig)

    reporte_final = f"""
REPORTE AUTOMÁTICO DE PROYECCIÓN TRM - WEB APP
----------------------------------------------

Fecha de consulta Colombia: {formato_fecha(fecha_hoy_colombia)}

TRM OFICIAL VIGENTE
-------------------
Fuente principal: {FUENTE_PRINCIPAL_TRM}
TRM vigente usada como base principal: {formato_cop(trm_vigente_hoy)}
Vigente desde: {formato_fecha(vigente_desde)}
Vigente hasta: {formato_fecha(vigente_hasta)}

VALIDACIÓN AUXILIAR
-------------------
Fuente auxiliar: {validacion_aux["fuente_auxiliar"]}
Valor auxiliar leído: {formato_cop(validacion_aux["valor_auxiliar"]) if validacion_aux["valor_auxiliar"] is not None else "NO DISPONIBLE"}
Estado validación auxiliar: {validacion_aux["estado_auxiliar"]}
Mensaje: {validacion_aux["mensaje_auxiliar"]}

CONTROL DE DATOS FUTUROS
------------------------
Registros futuros detectados: {len(df_futuros_bloqueados)}
Uso de registros futuros: NO
Valor futuro mostrado: NO

MODELO
------
Registros usados para entrenar: {mejor_modelo["registros_entrenamiento"]}
Mejor modelo: {mejor_modelo["modelo"]}
Error promedio histórico: {formato_cop(mejor_modelo["mae"])}

PROYECCIÓN SOLICITADA
---------------------
Fecha solicitada: {formato_fecha(fecha_solicitada)}
Fecha objetivo usada por el modelo: {formato_fecha(fecha_objetivo)}

Fecha base del modelo: {formato_fecha(fecha_base_modelo)}
TRM base del modelo: {formato_cop(trm_base_modelo)}

Días calendario hasta la fecha objetivo: {dias_calendario}
Días hábiles proyectados: {dias_habiles}

RESULTADO CENTRAL
-----------------
TRM estimada para la fecha objetivo:
{formato_cop(trm_estimacion_final)}

Cambio estimado frente a la TRM actual:
{formato_cop(cambio_total)}

Cambio porcentual estimado:
{formato_porcentaje(cambio_porcentual)}

Cambio promedio por día hábil proyectado:
{formato_cop(cambio_promedio_habil)}

Señal del modelo:
{senal}

Lectura:
{lectura}

ESCENARIOS
----------
Escenario bajo 80%: {formato_cop(rango_bajo_80)}
Escenario central: {formato_cop(trm_estimacion_final)}
Escenario alto 80%: {formato_cop(rango_alto_80)}

Escenario bajo 90%: {formato_cop(rango_bajo_90)}
Escenario alto 90%: {formato_cop(rango_alto_90)}

NIVEL DE CONFIANZA
------------------
{nivel_confianza}

ADVERTENCIA
-----------
Esta es una proyección estadística. No es una TRM oficial ni garantizada.
"""

    st.subheader("Descargas")

    st.download_button(
        "Descargar reporte TXT",
        data=reporte_final.encode("utf-8"),
        file_name=f"reporte_trm_{fecha_objetivo.strftime('%Y%m%d')}.txt",
        mime="text/plain"
    )

    st.download_button(
        "Descargar tabla de decisión CSV",
        data=df_decision.to_csv(index=False).encode("utf-8"),
        file_name=f"tabla_decision_trm_{fecha_objetivo.strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

    st.download_button(
        "Descargar escenarios CSV",
        data=df_escenarios.to_csv(index=False).encode("utf-8"),
        file_name=f"escenarios_trm_{fecha_objetivo.strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

    st.download_button(
        "Descargar proyección día a día CSV",
        data=df_proyeccion.to_csv(index=False).encode("utf-8"),
        file_name=f"proyeccion_dia_a_dia_trm_{fecha_objetivo.strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

st.divider()
st.caption(
    "La TRM oficial es diaria. Las variables de mercado ayudan al análisis, "
    "pero no convierten la proyección en una TRM oficial."
)
