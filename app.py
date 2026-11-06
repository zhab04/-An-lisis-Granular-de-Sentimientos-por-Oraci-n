import streamlit as st
from transformers import pipeline
import pandas as pd
import plotly.express as px
import spacy

st.set_page_config(page_title="Análisis Granular de Sentimientos", page_icon="📊", layout="wide")
st.title("📊 Análisis Granular de Sentimientos por Oración")
st.markdown("Esta aplicación desglosa textos complejos, identifica el sentimiento de cada oración y evalúa el nivel de confianza del modelo.")

@st.cache_resource
def load_spacy():
    return spacy.load("es_core_news_sm")

@st.cache_resource
def load_transformer():
    return pipeline("sentiment-analysis", model="pysentimiento/robertuito-sentiment-analysis")

with st.spinner("Cargando modelos de IA..."):
    nlp = load_spacy()
    analizador = load_transformer()

texto_defecto = (
    "La atención al cliente fue excelente y respondieron todas mis dudas al instante. "
    "Me encantó el diseño de la aplicación porque es muy intuitiva. "
    "La calidad del producto superó por completo mis expectativas. "
    "El paquete incluye la caja original y el manual de usuario en español. "
    "La entrega se realiza mediante un servicio de paquetería local. "
    "Pésimo servicio en la parte de facturación porque cobraron dos veces. "
    "La interfaz web se traba constantemente al intentar guardar los datos. "
    "Nadie en soporte técnico respondió a mis correos de reclamo."
)

texto = st.text_area("Ingresa un párrafo o comentario compuesto:", texto_defecto, height=150)

if st.button("Analizar Sentimientos"):
    if texto.strip():
        doc = nlp(texto)
        oraciones = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        
        if not oraciones:
            st.warning("No se detectaron oraciones válidas.")
        else:
            resultados_detalle = []
            traduccion = {'POS': 'Positivo', 'NEU': 'Neutral', 'NEG': 'Negativo'}
            
            for oracion in oraciones:
                res = analizador(oracion, truncation=True, max_length=128)[0]
                etiqueta_trad = traduccion.get(res['label'], res['label'])
                resultados_detalle.append({
                    "Oración": oracion,
                    "Sentimiento": etiqueta_trad,
                    "Confianza": res['score']
                })
            
            df_detalle = pd.DataFrame(resultados_detalle)
            
            resumen = df_detalle.groupby('Sentimiento').agg(
                Cantidad=('Sentimiento', 'count'),
                Confianza_Promedio=('Confianza', 'mean')
            ).reset_index()
            
            resumen['Confianza_Pct'] = (resumen['Confianza_Promedio'] * 100).round(2)
            
            st.subheader("📈 Resumen Gráfico de Sentimientos y Nivel de Confianza Promedio")
            
            col_m1, col_m2, col_m3 = st.columns(3)
            for cat, col in zip(['Positivo', 'Negativo', 'Neutral'], [col_m1, col_m2, col_m3]):
                sub_df = resumen[resumen['Sentimiento'] == cat]
                if not sub_df.empty:
                    cant = sub_df['Cantidad'].values[0]
                    conf = sub_df['Confianza_Pct'].values[0]
                    col.metric(f"Oraciones {cat}s", f"{cant}", f"Confianza prom: {conf}%")
                else:
                    col.metric(f"Oraciones {cat}s", "0", "N/A")
            
            fig = px.bar(
                resumen, 
                x="Sentimiento", 
                y="Cantidad", 
                color="Sentimiento",
                text=resumen.apply(lambda r: f"{r['Cantidad']} orac. ({r['Confianza_Pct']}%)", axis=1),
                color_discrete_map={'Positivo': '#2ecc71', 'Neutral': '#95a5a6', 'Negativo': '#e74c3c'},
                title="Cantidad de Oraciones por Sentimiento y su Confianza Promedio"
            )
            fig.update_traces(textposition='outside')
            fig.update_layout(yaxis_title="Número de Oraciones", xaxis_title="Sentimiento", yaxis_range=[0, max(resumen['Cantidad']) + 2])
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            st.subheader("📝 Desglose del Texto con Nivel de Confianza Individual")
            
            positis = df_detalle[df_detalle['Sentimiento'] == 'Positivo']
            negativs = df_detalle[df_detalle['Sentimiento'] == 'Negativo']
            neutras = df_detalle[df_detalle['Sentimiento'] == 'Neutral']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("### 🟢 Mensajes Positivos")
                if not positis.empty:
                    for _, row in positis.iterrows():
                        st.success(f"**\"{row['Oración']}\"**\n\n🎯 **Confianza:** `{row['Confianza'] * 100:.2f}%` ({row['Confianza']:.4f})")
                else:
                    st.info("No se detectaron fragmentos positivos.")
                    
            with col2:
                st.markdown("### 🔴 Mensajes Negativos")
                if not negativs.empty:
                    for _, row in negativs.iterrows():
                        st.error(f"**\"{row['Oración']}\"**\n\n🎯 **Confianza:** `{row['Confianza'] * 100:.2f}%` ({row['Confianza']:.4f})")
                else:
                    st.info("No se detectaron fragmentos negativos.")
                    
            with col3:
                st.markdown("### ⚪ Mensajes Neutrales")
                if not neutras.empty:
                    for _, row in neutras.iterrows():
                        st.warning(f"**\"{row['Oración']}\"**\n\n🎯 **Confianza:** `{row['Confianza'] * 100:.2f}%` ({row['Confianza']:.4f})")
                else:
                    st.info("No se detectaron fragmentos neutrales.")
    else:
        st.warning("Por favor, ingresa un texto válido.")
