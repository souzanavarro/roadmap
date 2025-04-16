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
            st.dataframe(frota_df)
            # Salvar no database local na pasta src/database
            os.makedirs(os.path.join("src", "database"), exist_ok=True)
            frota_db_path = os.path.join("src", "database", "database_frota.csv")
            frota_df.to_csv(frota_db_path, index=False)
            st.success(f"Frota salva no banco de dados local: {frota_db_path}")

    elif menu == "Dashboard Pedidos":
        st.header("Dashboard de Pedidos")
        pedidos_file = st.file_uploader("Envie a planilha de pedidos (CSV, XLSX, XLSM)", type=["csv", "xlsx", "xlsm"], key="pedidos")
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
            # Previsualização de mapa
            from streamlit_folium import folium_static
            import folium
            local_partida = [-23.0838, -47.1336]  # Coordenadas fixas de partida
            if 'Latitude' in pedidos_df.columns and 'Longitude' in pedidos_df.columns and not pedidos_df.empty:
                m = folium.Map(location=[pedidos_df['Latitude'].mean(), pedidos_df['Longitude'].mean()], zoom_start=10)
                for _, row in pedidos_df.iterrows():
                    folium.Marker([row['Latitude'], row['Longitude']], popup=row.get('Nome Cliente', '')).add_to(m)
                folium.Marker(local_partida, popup="Local de Partida", icon=folium.Icon(color='red')).add_to(m)
                st.subheader("Mapa dos Pedidos")
                folium_static(m)
            else:
                m = folium.Map(location=local_partida, zoom_start=10)
                folium.Marker(local_partida, popup="Local de Partida", icon=folium.Icon(color='red')).add_to(m)
                st.subheader("Mapa dos Pedidos")
                folium_static(m)
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
            # Previsualização de mapa
            if 'Latitude' in ia_df.columns and 'Longitude' in ia_df.columns:
                from streamlit_folium import folium_static
                import folium
                m = folium.Map(location=[ia_df['Latitude'].mean(), ia_df['Longitude'].mean()], zoom_start=10)
                for _, row in ia_df.iterrows():
                    folium.Marker([row['Latitude'], row['Longitude']], popup=row.get('Nome Cliente', '')).add_to(m)
                st.subheader("Mapa dos Pedidos Roteirizados")
                folium_static(m)
            else:
                st.info("Sua planilha precisa ter as colunas 'Latitude' e 'Longitude' para exibir o mapa.")

if __name__ == "__main__":
    main()