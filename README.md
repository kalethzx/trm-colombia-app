# Proyección TRM Colombia - Web App

Aplicación web en Streamlit para consultar TRM oficial, validar con fuente auxiliar y proyectar escenarios de TRM a una fecha futura.

## Archivos

- `app.py`: aplicación principal.
- `requirements.txt`: dependencias para Streamlit Community Cloud.

## Cómo probar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Cómo publicar en Streamlit Community Cloud

1. Crea un repositorio en GitHub.
2. Sube `app.py` y `requirements.txt`.
3. Entra a Streamlit Community Cloud.
4. Elige el repositorio, la rama y el archivo `app.py`.
5. Haz clic en Deploy.

## Notas

- La fuente principal de TRM es Datos Abiertos Colombia.
- Dólar-Colombia se usa solo como validación auxiliar gratuita.
- La app no usa Yahoo Finance.
- La TRM oficial es diaria; no es una cotización intradía.


## Corrección V2

Corrige lectura del modelo serializado con `StringIO` para evitar error `FileNotFoundError` en Streamlit Cloud.
