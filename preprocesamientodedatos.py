import pandas as pd
import numpy as np

# Configuración del archivo original
archivo_ug = 'REPORTE_RECORD_ESTUDIANTIL_ANONIMIZADO.xlsx'

try:
    # Usamos engine='openpyxl' por seguridad con archivos .xlsx modernos
    df = pd.read_excel(archivo_ug, sheet_name='Sheet1')
    print("Archivo UG cargado correctamente.")
except Exception as e:
    print(f"Error al cargar el Excel: {e}")
    exit()

# FUNCIÓN CORREGIDA: Convierte las comas en puntos
def corregir_notas(v):
    if isinstance(v, str): 
        return float(v.replace(',', '.'))
    return float(v)

df['nota_limpia'] = df['PROMEDIO'].apply(corregir_notas)

# Ajuste de asistencia por convalidaciones/movilidad
convalida_mask = df['GRUPO/PARALELO'].str.contains('MOVILIDAD|CONVALIDACION', na=False, case=False)
df['asistencia_final'] = df['ASISTENCIA']
df.loc[convalida_mask & (df['ESTADO'] == 'APROBADA'), 'asistencia_final'] = 100

# Identificar estudiantes activos en el ciclo 2025
activos_2025 = df[df['PERIODO'].str.contains('2025 - 2026', na=False)]['ESTUDIANTE'].unique()

# CREACIÓN DE LA TABLA MAESTRA
# Agrupamos por estudiante con los 3 nombres finales
dataset = df.groupby('ESTUDIANTE').agg(
    promediohistorico=('nota_limpia', 'mean'),
    promedioasistencia=('asistencia_final', 'mean'),
    **{'# de Semestre': ('NIVEL', 'max')}
).reset_index()

# LÓGICA DE VARIABLE OBJETIVO (DESERTOR)
# 1. Por ausencia en 2025
dataset['Desertor'] = dataset['ESTUDIANTE'].apply(lambda x: 0 if x in activos_2025 else 1)

# 2. Por reglamento: 3ra matrícula reprobada
perdio_3ra = df[(df['NO. VEZ'] == 3) & (df['ESTADO'] == 'REPROBADA')]['ESTUDIANTE'].unique()
dataset.loc[dataset['ESTUDIANTE'].isin(perdio_3ra), 'Desertor'] = 1

# 3. Por reglamento: Asistencia insuficiente (< 70)
dataset.loc[dataset['promedioasistencia'] < 70, 'Desertor'] = 1

# Guardar resultado
dataset.to_csv('dataset.csv', index=False)
print("✅ 'dataset.csv' generado con éxito.")