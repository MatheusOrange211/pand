import streamlit as st
from pand_system.agent import run_agent_analysis
from PIL import Image
import io

st.set_page_config(page_title="PAND System (Groq Edition)", layout="wide")

st.title("🛡️ PAND - Sistema Inteligente de Manutenção Solar")
st.markdown("**Core:** Llama 3 via Groq | **Arquitetura:** Multi-Agentes")

# Sidebar para configurações
with st.sidebar:
    painel_id = st.text_input("ID do Painel", value="painel_01")
    uploaded_file = st.file_uploader("Upload da Imagem do Painel", type=["jpg", "png", "jpeg"])

if uploaded_file and painel_id:
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(uploaded_file, caption="Imagem Carregada", use_column_width=True)
    
    with col2:
        if st.button("🚀 Iniciar Análise Diagnóstica"):
            with st.spinner("🤖 Os 4 Agentes estão trabalhando..."):
                # Converte imagem para bytes
                bytes_data = uploaded_file.getvalue()
                
                # Chama o orquestrador
                resultado = run_agent_analysis(bytes_data, painel_id)
                
                st.success("Análise Concluída!")
                st.markdown("---")
                st.markdown(resultado)