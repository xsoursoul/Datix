import pandas as pd
import unicodedata
from pathlib import Path

# ── Rutas ────────────────────────────────────────────────────────────────────
RAW_DIR = Path("data/raw")
TRF_DIR = Path("data/trf")
TRF_DIR.mkdir(parents=True, exist_ok=True)

# ── Helpers ──────────────────────────────────────────────────────────────────

def parse_fecha(series: pd.Series) -> pd.Series:
    """Parsea fechas con múltiples formatos. Fechas inválidas → NaT."""
    formatos = ["%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d"]
    resultado = pd.Series([pd.NaT] * len(series), index=series.index)
    sin_parsear = series.copy()

    for fmt in formatos:
        mascara = sin_parsear.notna()
        intentos = pd.to_datetime(sin_parsear[mascara], format=fmt, errors="coerce")
        exito = intentos.notna()
        resultado[sin_parsear.index[mascara][exito]] = intentos[exito].values
        sin_parsear[sin_parsear.index[mascara][exito]] = pd.NaT

    return resultado


def normalizar_texto(series: pd.Series) -> pd.Series:
    """Strip + title case. Conserva NaN."""
    return series.str.strip().str.title()


def unificar_marca(series: pd.Series) -> pd.Series:
    """Elimina espacios internos para unificar variantes (ej: 'Urban Wear' → 'UrbanWear')."""
    return series.str.strip().str.replace(r"\s+", "", regex=True).str.title()


# ── inventario.csv ───────────────────────────────────────────────────────────

def remove_accents(series: pd.Series) -> pd.Series:
    """Elimina tildes y diacríticos de una columna de texto."""
    def _strip(val):
        if pd.isna(val):
            return val
        return "".join(
            c for c in unicodedata.normalize("NFD", str(val))
            if unicodedata.category(c) != "Mn"
        )
    return series.map(_strip)


def limpiar_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica remove_accents a todas las columnas de tipo string/object."""
    for col in df.select_dtypes(include="object").columns:
        df[col] = remove_accents(df[col])
    return df



def limpiar_inventario() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "inventario.csv")

    df["nombre_producto"] = normalizar_texto(df["nombre_producto"])
    df["marca"]           = unificar_marca(df["marca"])
    df["color"]           = normalizar_texto(df["color"])
    df["categoria"]       = normalizar_texto(df["categoria"])
    df["fecha_ingreso"]   = parse_fecha(df["fecha_ingreso"])
    df["fecha_ingreso"]   = pd.to_datetime(df["fecha_ingreso"], errors="coerce")
    df["fecha_ingreso"]   = df["fecha_ingreso"].fillna(pd.Timestamp("1900-01-01"))

    df = limpiar_strings(df)
    df.to_csv(TRF_DIR / "inventario_trf.csv", index=False)
    print(f"[inventario] {len(df)} filas guardadas → {TRF_DIR / 'inventario_trf.csv'}")
    return df


# ── ventas.csv ───────────────────────────────────────────────────────────────

MEDIO_PAGO_MAP = {
    "debito":  "Debito",
    "débito":  "Debito",
    "credito": "Credito",
    "crédito": "Credito",
}

def normalizar_medio_pago(series: pd.Series) -> pd.Series:
    """Unifica variantes con/sin tilde. Valores no mapeados → title case."""
    lower = series.str.strip().str.lower()
    return lower.map(MEDIO_PAGO_MAP).fillna(series.str.strip().str.title())


def limpiar_ventas() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "ventas.csv")

    df["fecha"]       = parse_fecha(df["fecha"])
    df["cliente"]     = normalizar_texto(df["cliente"])
    df["rut_cliente"] = df["rut_cliente"].str.strip().str.upper()
    df["medio_pago"]  = normalizar_medio_pago(df["medio_pago"])
    df["sucursal"]    = normalizar_texto(df["sucursal"])

    # Nulos en cliente, rut_cliente, precio_unitario: pendiente de decisión

    df = limpiar_strings(df)
    df.to_csv(TRF_DIR / "ventas_trf.csv", index=False)
    print(f"[ventas]     {len(df)} filas guardadas → {TRF_DIR / 'ventas_trf.csv'}")
    return df


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    limpiar_inventario()
    limpiar_ventas()
