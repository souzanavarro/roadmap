import streamlit as st
import pandas as pd
import os
from geocode import obter_coordenadas_com_fallback

def dashboard_ia():
    st.header("Dashboard IA - Base de Roteirizações")
    ia_file = st.file_uploader("Envie a planilha de pedidos já roteirizada (CSV, XLSX, XLSM)", type=["csv", "xlsx", "xlsm"], key="ia")
    if ia_file:
        if ia_file.name.endswith(".csv"):
            ia_df = pd.read_csv(ia_file)
        else:
            ia_df = pd.read_excel(ia_file, engine="openpyxl")
        # Formar o endereço completo
        if 'Endereço Completo' not in ia_df.columns:
            ia_df['Endereço Completo'] = ia_df['Endereço de Entrega'] + ', ' + ia_df['Bairro de Entrega'] + ', ' + ia_df['Cidade de Entrega']
        # Obter coordenadas se não existirem
        if 'Latitude' not in ia_df.columns or 'Longitude' not in ia_df.columns or ia_df['Latitude'].isnull().any() or ia_df['Longitude'].isnull().any():
            st.info("Obtendo coordenadas para os endereços...")
            coordenadas_salvas = {}
            api_key = "6f522c67add14152926990afbe127384"
            def get_coords(row):
                lat, lon = obter_coordenadas_com_fallback(row['Endereço Completo'], coordenadas_salvas, api_key)
                return pd.Series({'Latitude': lat, 'Longitude': lon})
            coords = ia_df.apply(get_coords, axis=1)
            ia_df['Latitude'] = coords['Latitude']
            ia_df['Longitude'] = coords['Longitude']
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
