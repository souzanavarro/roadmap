import streamlit as st
import pandas as pd
import os

def dashboard_frota():
    st.header("Dashboard da Frota")
    frota_db_path = os.path.join("src", "database", "database_frota.csv")
    # Carregar frota existente, se houver
    if os.path.exists(frota_db_path):
        frota_df = pd.read_csv(frota_db_path)
    else:
        frota_df = pd.DataFrame()
    # Verificar se as colunas obrigatórias estão presentes
    colunas_obrigatorias = ['Placa', 'Transportador', 'Descrição Veículo', 'Capac. Cx', 'Capac. Kg', 'Disponível']
    if not all(col in frota_df.columns for col in colunas_obrigatorias):
        st.error(f"A planilha da frota está faltando as seguintes colunas obrigatórias: {', '.join([col for col in colunas_obrigatorias if col not in frota_df.columns])}")
        return
    frota_file = st.file_uploader("Envie a planilha da frota (CSV, XLSX, XLSM)", type=["csv", "xlsx", "xlsm"], key="frota")
    if frota_file:
        if frota_file.name.endswith(".csv"):
            new_frota_df = pd.read_csv(frota_file)
        else:
            new_frota_df = pd.read_excel(frota_file, engine="openpyxl")
        # Remover placas indesejadas
        placas_excluir = [
            "FLB1111", "FLB2222", "FLB3333", "FLB4444", "FLB5555",
            "FLB6666", "FLB7777", "FLB8888", "FLB9999", "HFU1B60", "CYN1819"
        ]
        if 'Placa' in new_frota_df.columns:
            new_frota_df = new_frota_df[~new_frota_df['Placa'].isin(placas_excluir)]
        frota_df = new_frota_df.copy()
        frota_df.to_csv(frota_db_path, index=False)
        st.success(f"Frota salva no banco de dados local: {frota_db_path}")
    # Permitir edição da planilha da frota
    if not frota_df.empty:
        edited_frota_df = st.data_editor(frota_df, num_rows="dynamic")
        if st.button("Salvar Frota Editada"):
            edited_frota_df.to_csv(frota_db_path, index=False)
            st.success(f"Frota editada salva no banco de dados local: {frota_db_path}")
    else:
        st.info("Nenhuma frota cadastrada. Faça upload de uma planilha de frota.")
    # Botão para limpar frota
    if st.button("Limpar Frota"):
        if os.path.exists(frota_db_path):
            os.remove(frota_db_path)
            st.success("Frota removida com sucesso!")
        else:
            st.info("Nenhuma frota para remover.")
