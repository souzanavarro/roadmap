import streamlit as st
import pandas as pd
import os

def main():
    st.set_page_config(page_title="Roteirizador de Entregas", layout="wide")
    st.sidebar.title("Menu")
    menu = st.sidebar.radio("Escolha uma opção:", ("Upload Frota", "Upload Pedidos"))

    if menu == "Upload Frota":
        st.header("Upload da Planilha de Frota")
        frota_file = st.file_uploader("Envie a planilha da frota (CSV)", type=["csv"], key="frota")
        if frota_file:
            frota_df = pd.read_csv(frota_file)
            st.dataframe(frota_df)
            # Salvar arquivo enviado (opcional)
            save_path = os.path.join("..", "data", "frota_upload.csv")
            frota_df.to_csv(save_path, index=False)
            st.success(f"Arquivo salvo em {save_path}")

    elif menu == "Upload Pedidos":
        st.header("Upload da Planilha de Pedidos")
        pedidos_file = st.file_uploader("Envie a planilha de pedidos (CSV)", type=["csv"], key="pedidos")
        if pedidos_file:
            pedidos_df = pd.read_csv(pedidos_file)
            st.dataframe(pedidos_df)
            # Salvar arquivo enviado (opcional)
            save_path = os.path.join("..", "data", "pedidos_upload.csv")
            pedidos_df.to_csv(save_path, index=False)
            st.success(f"Arquivo salvo em {save_path}")

if __name__ == "__main__":
    main()
