from __future__ import annotations

import io
from typing import Callable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError
from scipy.interpolate import PchipInterpolator


st.set_page_config(
    page_title="Laboratorio de transformación de intensidades",
    page_icon="🖼️",
    layout="wide",
)

NIVELES = np.arange(256, dtype=float)


def leer_imagen_gris(archivo) -> np.ndarray:
    """Lee una imagen cargada y la convierte a escala de grises de 8 bits."""
    imagen_pil = Image.open(archivo)
    imagen_pil = ImageOps.exif_transpose(imagen_pil)
    return np.asarray(imagen_pil.convert("L"), dtype=np.uint8)


def construir_aproximacion(
    cdf: np.ndarray,
    numero_puntos: int,
    metodo: str,
    numero_terminos_fourier: int = 5,
) -> tuple[Callable[[np.ndarray], np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """Construye una aproximación de la CDF y devuelve evaluador, curva y puntos."""
    puntos_x = np.linspace(0.0, 255.0, int(numero_puntos))
    puntos_y = np.interp(puntos_x, NIVELES, cdf)

    if metodo == "PCHIP (cúbica por tramos)":
        interpolante = PchipInterpolator(puntos_x, puntos_y, extrapolate=False)

        def evaluar(x):
            x = np.asarray(x, dtype=float)
            x_seguro = np.clip(x, 0.0, 255.0)
            return np.asarray(interpolante(x_seguro), dtype=float)

    elif metodo == "Lineal":
        def evaluar(x):
            x = np.asarray(x, dtype=float)
            return np.interp(np.clip(x, 0.0, 255.0), puntos_x, puntos_y)

    elif metodo == "Fourier trigonométrica":
        # Ajuste por mínimos cuadrados de una serie trigonométrica sobre [0, 255].
        # El uso de pocos términos reduce la complejidad, aunque pueden aparecer
        # oscilaciones y sobrepasos cerca de los extremos.
        terminos = max(1, int(numero_terminos_fourier))

        def matriz_fourier(x):
            x = np.asarray(x, dtype=float).reshape(-1)
            t = 2.0 * np.pi * np.clip(x, 0.0, 255.0) / 255.0
            columnas = [np.ones_like(t)]
            for k in range(1, terminos + 1):
                columnas.append(np.cos(k * t))
                columnas.append(np.sin(k * t))
            return np.column_stack(columnas)

        matriz_puntos = matriz_fourier(puntos_x)
        coeficientes, *_ = np.linalg.lstsq(matriz_puntos, puntos_y, rcond=None)

        def evaluar(x):
            return matriz_fourier(np.asarray(x, dtype=float)) @ coeficientes

    else:
        raise ValueError(f"Método no reconocido: {metodo}")

    curva_aproximada = np.asarray(evaluar(NIVELES), dtype=float)
    return evaluar, curva_aproximada, puntos_x, puntos_y


def calcular_metricas(imagen: np.ndarray) -> dict[str, float]:
    """Calcula métricas descriptivas básicas de una imagen en escala de grises."""
    arr = np.asarray(imagen, dtype=np.uint8)
    hist = np.bincount(arr.ravel(), minlength=256)
    probabilidades = hist / arr.size
    probabilidades_positivas = probabilidades[probabilidades > 0]
    entropia = -np.sum(probabilidades_positivas * np.log2(probabilidades_positivas))
    p1, p99 = np.percentile(arr, [1, 99])

    return {
        "Media": float(np.mean(arr)),
        "Desviación estándar": float(np.std(arr)),
        "Mínimo": int(arr.min()),
        "Máximo": int(arr.max()),
        "Rango dinámico": int(arr.max() - arr.min()),
        "Rango P1–P99": float(p99 - p1),
        "Entropía (bits)": float(entropia),
        "Píxeles en 0 (%)": float(100 * np.mean(arr == 0)),
        "Píxeles en 255 (%)": float(100 * np.mean(arr == 255)),
    }


def figura_a_png(fig) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return buffer.getvalue()


def graficar_histogramas(resultados: dict[str, np.ndarray]) -> bytes:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for nombre, img in resultados.items():
        hist = np.bincount(img.ravel(), minlength=256)
        ax.plot(NIVELES, hist / img.size, linewidth=1.5, label=nombre)
    ax.set_title("Comparación de histogramas")
    ax.set_xlabel("Nivel de intensidad")
    ax.set_ylabel("Frecuencia relativa")
    ax.set_xlim(0, 255)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return figura_a_png(fig)


def graficar_cdfs(resultados: dict[str, np.ndarray]) -> bytes:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for nombre, img in resultados.items():
        hist = np.bincount(img.ravel(), minlength=256)
        cdf_img = np.cumsum(hist) / img.size
        ax.plot(NIVELES, cdf_img, linewidth=1.7, label=nombre)
    ax.set_title("Comparación de funciones de distribución acumulada")
    ax.set_xlabel("Nivel de intensidad")
    ax.set_ylabel("Probabilidad acumulada")
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return figura_a_png(fig)


st.title("Laboratorio de transformación de intensidades")
st.write(
    "Explora cómo una transformación de cuatro parámetros modifica una aproximación "
    "de la CDF de una imagen en escala de grises. Los resultados se actualizan al "
    "cambiar los controles."
)

with st.sidebar:
    st.header("1. Imagen")
    archivo = st.file_uploader(
        "Carga una imagen",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"],
        help="La imagen se convierte a escala de grises para realizar el análisis.",
    )

    st.divider()
    st.header("2. Aproximación de la CDF")
    metodo = st.selectbox(
        "Método de aproximación",
        ["PCHIP (cúbica por tramos)", "Lineal", "Fourier trigonométrica"],
        index=0,
        help=(
            "PCHIP conserva la monotonía de los puntos de control. La serie de Fourier "
            "puede presentar oscilaciones y sobrepasos; esto también puede ser objeto de análisis."
        ),
    )
    numero_puntos = st.slider(
        "Número de puntos de control",
        min_value=2,
        max_value=256,
        value=32,
        step=1,
        help="Se eligen puntos distribuidos uniformemente entre las intensidades 0 y 255.",
    )
    numero_terminos = 5
    if metodo == "Fourier trigonométrica":
        max_terminos = max(1, min(30, (numero_puntos - 1) // 2))
        numero_terminos = st.slider(
            "Número de términos de Fourier",
            min_value=1,
            max_value=max_terminos,
            value=min(5, max_terminos),
            step=1,
            help="Más términos permiten mayor complejidad, pero también pueden aumentar las oscilaciones.",
        )

    st.divider()
    st.header("3. Parámetros de transformación")

    # Guardar los valores de los controles en session_state permite reiniciarlos
    # antes de que Streamlit vuelva a construir los widgets.
    valores_iniciales = {
        "param_A": 1.0,
        "param_B": 0.0,
        "param_C": 1.0,
        "param_D": 0.0,
    }
    for clave, valor_inicial in valores_iniciales.items():
        if clave not in st.session_state:
            st.session_state[clave] = valor_inicial

    if st.button("Restablecer parámetros neutros", use_container_width=True):
        for clave, valor_inicial in valores_iniciales.items():
            st.session_state[clave] = valor_inicial
        st.rerun()

    A = st.slider(
        "A — escala horizontal",
        min_value=-3.0,
        max_value=3.0,
        step=0.05,
        key="param_A",
        help="Multiplica la variable dentro de la función: A·r + B.",
    )
    B = st.slider(
        "B — desplazamiento horizontal",
        min_value=-255.0,
        max_value=255.0,
        step=1.0,
        key="param_B",
        help="Se suma a la variable antes de evaluar la función.",
    )
    C = st.slider(
        "C — escala vertical",
        min_value=-3.0,
        max_value=3.0,
        step=0.05,
        key="param_C",
        help="Multiplica el valor de la función transformada.",
    )
    D = st.slider(
        "D — desplazamiento vertical",
        min_value=-1.0,
        max_value=1.0,
        step=0.01,
        key="param_D",
        help="Se suma al valor de la función transformada.",
    )

if archivo is None:
    st.info("Carga una imagen desde el panel izquierdo para comenzar.")
    st.markdown(
        "### Flujo del laboratorio\n"
        "1. Conversión a escala de grises.\n"
        "2. Cálculo del histograma, la CDF y las métricas.\n"
        "3. Aproximación de la CDF mediante puntos de control.\n"
        "4. Transformación de la función con los parámetros A, B, C y D.\n"
        "5. Aplicación del mapa de intensidades y comparación con HE y CLAHE."
    )
    st.stop()

try:
    imagen = leer_imagen_gris(archivo)
except (UnidentifiedImageError, OSError, ValueError) as exc:
    st.error(f"No se pudo abrir la imagen. Verifica el archivo. Detalle: {exc}")
    st.stop()

if imagen.size == 0:
    st.error("La imagen está vacía o no se pudo leer correctamente.")
    st.stop()

niveles_enteros = np.arange(256)
hist_original = np.bincount(imagen.ravel(), minlength=256)
cdf_original = np.cumsum(hist_original) / imagen.size

evaluar, cdf_aproximada, puntos_x, puntos_y = construir_aproximacion(
    cdf_original,
    numero_puntos,
    metodo,
    numero_terminos,
)

# Modelo propuesto: G(r) = C * F_aprox(A*r + B) + D.
# El mapa usa la diferencia G(r)-F_aprox(r), de modo que los parámetros
# neutros (A=1, B=0, C=1, D=0) producen exactamente T(r)=r.
argumento = np.clip(A * NIVELES + B, 0.0, 255.0)
funcion_transformada = C * np.asarray(evaluar(argumento), dtype=float) + D
diferencia_funcional = funcion_transformada - cdf_aproximada
mapa_continuo = NIVELES + 255.0 * diferencia_funcional
mapa_intensidades = np.rint(np.clip(mapa_continuo, 0.0, 255.0)).astype(np.uint8)
imagen_modelo = mapa_intensidades[imagen]

imagen_he = cv2.equalizeHist(imagen)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
imagen_clahe = clahe.apply(imagen)

resultados = {
    "Original": imagen,
    "Modelo propuesto": imagen_modelo,
    "Ecualización clásica (HE)": imagen_he,
    "CLAHE": imagen_clahe,
}

# Calidad de aproximación y estado del modelo.
error_medio_aprox = float(np.mean(np.abs(cdf_original - cdf_aproximada)))
error_max_aprox = float(np.max(np.abs(cdf_original - cdf_aproximada)))
imagen_identica = bool(np.array_equal(imagen, imagen_modelo))
pixeles_modificados = int(np.count_nonzero(imagen != imagen_modelo))

st.caption(
    f"Imagen en escala de grises · {imagen.shape[1]} × {imagen.shape[0]} píxeles · "
    f"Aproximación: {metodo} · {numero_puntos} puntos de control"
)

pestanas = st.tabs([
    "1. Imagen y distribución",
    "2. Aproximación",
    "3. Transformación",
    "4. Comparación final",
])

with pestanas[0]:
    col_img, col_hist, col_cdf = st.columns([1.1, 1.3, 1.3])
    with col_img:
        st.subheader("Imagen original")
        st.image(imagen, clamp=True, use_container_width=True)
    with col_hist:
        st.subheader("Histograma original")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.bar(niveles_enteros, hist_original, width=1.0, color="gray")
        ax.set_xlabel("Intensidad")
        ax.set_ylabel("Frecuencia")
        ax.set_xlim(0, 255)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with col_cdf:
        st.subheader("CDF original")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.plot(niveles_enteros, cdf_original, linewidth=2)
        ax.set_xlabel("Intensidad")
        ax.set_ylabel("Probabilidad acumulada")
        ax.set_xlim(0, 255)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.subheader("Métricas de la imagen original")
    metricas_original = pd.DataFrame(
        [calcular_metricas(imagen)], index=["Original"]
    ).T
    st.dataframe(metricas_original.round(4), use_container_width=True)

with pestanas[1]:
    st.subheader("CDF empírica e interpolación/aproximación")
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(niveles_enteros, cdf_original, label="CDF empírica", linewidth=2)
    ax.plot(niveles_enteros, cdf_aproximada, label=f"Aproximación: {metodo}", linewidth=2)
    ax.scatter(puntos_x, puntos_y, s=25, zorder=3, label="Puntos de control")
    ax.set_xlabel("Intensidad")
    ax.set_ylabel("Probabilidad acumulada")
    ax.set_xlim(0, 255)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    m1, m2 = st.columns(2)
    m1.metric("Error absoluto medio", f"{error_medio_aprox:.6f}")
    m2.metric("Error absoluto máximo", f"{error_max_aprox:.6f}")
    if metodo == "Fourier trigonométrica":
        st.warning(
            "La aproximación trigonométrica puede oscilar o salir del intervalo [0, 1]. "
            "No se recorta la curva aproximada para que puedas observar esos efectos; "
            "las intensidades finales sí se limitan a [0, 255]."
        )
    else:
        st.caption(
            "PCHIP y la interpolación lineal se construyen a partir de los puntos seleccionados. "
            "Cambiar el número de puntos modifica la fidelidad de la aproximación."
        )

with pestanas[2]:
    st.subheader("Función transformada y mapa de intensidades")
    st.latex(r"G(r)=C\,\widehat F(Ar+B)+D")
    st.latex(r"T(r)=r+255\,[G(r)-\widehat F(r)]")
    st.write(
        "La segunda expresión usa la diferencia entre la función transformada y la aproximación "
        "original. Así, con A=1, B=0, C=1 y D=0, el mapa es la identidad y no modifica la imagen."
    )

    col_funcion, col_mapa = st.columns(2)
    with col_funcion:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(NIVELES, cdf_aproximada, label="Función aproximada original", linewidth=2)
        ax.plot(NIVELES, funcion_transformada, label="G(r): función transformada", linewidth=2)
        ax.set_title("Transformación de la función")
        ax.set_xlabel("Variable de entrada r")
        ax.set_ylabel("Valor de la función")
        ax.set_xlim(0, 255)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with col_mapa:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(NIVELES, mapa_continuo, label="Mapa T(r) antes del redondeo", linewidth=2)
        ax.plot(NIVELES, NIVELES, "--", label="Identidad T(r)=r", linewidth=1.7)
        ax.set_title("Mapa aplicado a los píxeles")
        ax.set_xlabel("Intensidad original r")
        ax.set_ylabel("Intensidad resultante T(r)")
        ax.set_xlim(0, 255)
        ax.set_ylim(-10, 265)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Píxeles modificados", f"{pixeles_modificados:,}")
    col_b.metric("Porcentaje modificado", f"{100 * pixeles_modificados / imagen.size:.2f}%")
    col_c.metric("¿Imagen idéntica?", "Sí" if imagen_identica else "No")

    st.subheader("Imagen obtenida con el modelo")
    st.image(imagen_modelo, clamp=True, use_container_width=True)
    salida = io.BytesIO()
    Image.fromarray(imagen_modelo).save(salida, format="PNG")
    st.download_button(
        "Descargar imagen transformada (PNG)",
        data=salida.getvalue(),
        file_name="imagen_transformada.png",
        mime="image/png",
    )

with pestanas[3]:
    st.subheader("Comparación visual")
    columnas = st.columns(4)
    for columna, (nombre, img) in zip(columnas, resultados.items()):
        with columna:
            st.markdown(f"**{nombre}**")
            st.image(img, clamp=True, use_container_width=True)

    st.subheader("Comparación de histogramas")
    st.image(graficar_histogramas(resultados), use_container_width=True)

    st.subheader("Comparación de CDF")
    st.image(graficar_cdfs(resultados), use_container_width=True)

    st.subheader("Tabla comparativa de métricas")
    tabla = pd.DataFrame(
        {nombre: calcular_metricas(img) for nombre, img in resultados.items()}
    ).T
    st.dataframe(tabla.round(4), use_container_width=True)

    metrica_grafico = st.selectbox(
        "Métrica para comparar gráficamente",
        [
            "Media",
            "Desviación estándar",
            "Rango dinámico",
            "Rango P1–P99",
            "Entropía (bits)",
            "Píxeles en 0 (%)",
            "Píxeles en 255 (%)",
        ],
    )
    fig, ax = plt.subplots(figsize=(9, 4))
    valores = tabla[metrica_grafico]
    ax.bar(valores.index, valores.values)
    ax.set_title(f"Comparación de la métrica: {metrica_grafico}")
    ax.set_ylabel(metrica_grafico)
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    csv = tabla.to_csv(index=True).encode("utf-8-sig")
    st.download_button(
        "Descargar métricas como CSV",
        data=csv,
        file_name="comparacion_metricas.csv",
        mime="text/csv",
    )

st.divider()
st.caption(
    "Herramienta experimental para explorar transformaciones de funciones. "
    "No se garantiza que todos los parámetros produzcan mapas monótonos ni mejoras visuales. "
    "Las métricas numéricas complementan, pero no sustituyen, la evaluación visual."
)
