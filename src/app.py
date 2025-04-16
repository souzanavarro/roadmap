import streamlit as st
import pandas as pd
import os

def main():
    st.set_page_config(page_title="Roteirizador de Entregas", layout="wide")
    st.sidebar.title("Menu")
    menu = st.sidebar.radio("Escolha uma opção:", ("Dashboard Frota", "Dashboard Pedidos", "Dashboard IA"))

    if menu == "Dashboard Frota":
        st.header("Dashboard da Frota")
        frota_file = st.file_uploader("Envie a planilha da frota (CSV, XLSX, XLSM)", type=["csv", "xlsx", "xlsm"], key="frota")
        if frota_file:
            if frota_file.name.endswith(".csv"):
                frota_df = pd.read_csv(frota_file)
            else:
                frota_df = pd.read_excel(frota_file, engine="openpyxl")
            # Remover placas indesejadas
            placas_excluir = [
                "FLB1111", "FLB2222", "FLB3333", "FLB4444", "FLB5555",
                "FLB6666", "FLB7777", "FLB8888", "FLB9999", "HFU1B60", "CYN1819"
            ]
            if 'Placa' in frota_df.columns:
                frota_df = frota_df[~frota_df['Placa'].isin(placas_excluir)]
            st.dataframe(frota_df)
            # Permitir edição da planilha da frota
            edited_frota_df = st.data_editor(frota_df, num_rows="dynamic")
            # Salvar no database local na pasta src/database
            os.makedirs(os.path.join("src", "database"), exist_ok=True)
            frota_db_path = os.path.join("src", "database", "database_frota.csv")
            if st.button("Salvar Frota Editada"):
                edited_frota_df.to_csv(frota_db_path, index=False)
                st.success(f"Frota editada salva no banco de dados local: {frota_db_path}")

    elif menu == "Dashboard Pedidos":
        st.header("Dashboard de Pedidos")
        pedidos_file = st.file_uploader("Envie a planilha de pedidos (CSV, XLSX, XLSM)", type=["csv", "xlsx", "xlsm"], key="pedidos")
        # Carregar frota disponível
        frota_db_path = os.path.join("src", "database", "database_frota.csv")
        frota_disponivel = None
        if os.path.exists(frota_db_path):
            frota_disponivel = pd.read_csv(frota_db_path)
            if 'Disponível' in frota_disponivel.columns:
                frota_disponivel = frota_disponivel[frota_disponivel['Disponível'].str.lower() == 'sim']
        else:
            st.warning("Nenhuma frota cadastrada. Cadastre a frota antes de roterizar pedidos.")
        if pedidos_file:
            if pedidos_file.name.endswith(".csv"):
                pedidos_df = pd.read_csv(pedidos_file)
            else:
                pedidos_df = pd.read_excel(pedidos_file, engine="openpyxl")
            st.dataframe(pedidos_df)
            # Salvar no database local na pasta src/database
            os.makedirs(os.path.join("src", "database"), exist_ok=True)
            pedidos_db_path = os.path.join("src", "database", "database_pedidos.csv")
            pedidos_df.to_csv(pedidos_db_path, index=False)
            st.success(f"Pedidos salvos no banco de dados local: {pedidos_db_path}")
            df_map = pedidos_df
            # Checagem de frota disponível
            if frota_disponivel is not None and frota_disponivel.empty:
                st.error("Não há veículos disponíveis na frota para roterização!")
            elif frota_disponivel is None:
                st.warning("Não foi possível verificar a disponibilidade da frota.")
        else:
            # Tenta carregar o último database salvo
            pedidos_db_path = os.path.join("src", "database", "database_pedidos.csv")
            if os.path.exists(pedidos_db_path):
                df_map = pd.read_csv(pedidos_db_path)
            else:
                df_map = pd.DataFrame()
        # Previsualização de mapa sempre ativa
        from streamlit_folium import folium_static
        import folium
        local_partida = [-23.0838, -47.1336]  # Coordenadas fixas de partida
        if not df_map.empty and 'Latitude' in df_map.columns and 'Longitude' in df_map.columns:
            st.subheader("Mapa dos Pedidos")
            m = folium.Map(location=[df_map['Latitude'].mean(), df_map['Longitude'].mean()], zoom_start=10)
            for _, row in df_map.iterrows():
                folium.Marker([row['Latitude'], row['Longitude']], popup=row.get('Nome Cliente', '')).add_to(m)
            folium.Marker(local_partida, popup="Local de Partida", icon=folium.Icon(color='red')).add_to(m)
            folium_static(m, width=1200, height=500)
        else:
            st.subheader("Mapa dos Pedidos")
            m = folium.Map(location=local_partida, zoom_start=10)
            folium.Marker(local_partida, popup="Local de Partida", icon=folium.Icon(color='red')).add_to(m)
            folium_static(m, width=1200, height=500)
            st.info("Sua planilha precisa ter as colunas 'Latitude' e 'Longitude' para exibir os pedidos no mapa.")

    elif menu == "Dashboard IA":
        st.header("Dashboard IA - Base de Roteirizações")
        ia_file = st.file_uploader("Envie a planilha de pedidos já roteirizada (CSV, XLSX, XLSM)", type=["csv", "xlsx", "xlsm"], key="ia")
        if ia_file:
            if ia_file.name.endswith(".csv"):
                ia_df = pd.read_csv(ia_file)
            else:
                ia_df = pd.read_excel(ia_file, engine="openpyxl")
            st.dataframe(ia_df)
            # Salvar no database local na pasta src/database
            os.makedirs(os.path.join("src", "database"), exist_ok=True)
            ia_db_path = os.path.join("src", "database", "database_ia.csv")
            ia_df.to_csv(ia_db_path, index=False)
            st.success(f"Base IA salva no banco de dados local: {ia_db_path}")
            df_map = ia_df
        else:
            ia_db_path = os.path.join("src", "database", "database_ia.csv")
            if os.path.exists(ia_db_path):
                df_map = pd.read_csv(ia_db_path)
            else:
                df_map = pd.DataFrame()
        # Previsualização de mapa sempre ativa
        from streamlit_folium import folium_static
        import folium
        local_partida = [-23.0838, -47.1336]
        if not df_map.empty and 'Latitude' in df_map.columns and 'Longitude' in df_map.columns:
            st.subheader("Mapa dos Pedidos Roteirizados")
            m = folium.Map(location=[df_map['Latitude'].mean(), df_map['Longitude'].mean()], zoom_start=10)
            for _, row in df_map.iterrows():
                folium.Marker([row['Latitude'], row['Longitude']], popup=row.get('Nome Cliente', '')).add_to(m)
            folium.Marker(local_partida, popup="Local de Partida", icon=folium.Icon(color='red')).add_to(m)
            folium_static(m, width=1200, height=500)
        else:
            st.subheader("Mapa dos Pedidos Roteirizados")
            m = folium.Map(location=local_partida, zoom_start=10)
            folium.Marker(local_partida, popup="Local de Partida", icon=folium.Icon(color='red')).add_to(m)
            folium_static(m, width=1200, height=500)
            st.info("Sua planilha precisa ter as colunas 'Latitude' e 'Longitude' para exibir o mapa.")

if __name__ == "__main__":
    main()
