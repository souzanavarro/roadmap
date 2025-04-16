import streamlit as st
import pandas as pd
import os
from streamlit_folium import folium_static
import folium

def dashboard_routing():
    st.header("Dashboard Routing - Roteirização de Pedidos")
    # Carregar pedidos e frota do banco de dados local
    pedidos_db_path = os.path.join("src", "database", "database_pedidos.csv")
    frota_db_path = os.path.join("src", "database", "database_frota.csv")
    if os.path.exists(pedidos_db_path):
        pedidos_df = pd.read_csv(pedidos_db_path)
    else:
        pedidos_df = pd.DataFrame()
    if os.path.exists(frota_db_path):
        frota_df = pd.read_csv(frota_db_path)
    else:
        frota_df = pd.DataFrame()
    # Exibir dados carregados
    st.subheader("Pedidos para Roteirizar")
    st.dataframe(pedidos_df)
    st.subheader("Frota Disponível")
    st.dataframe(frota_df)
    # Previsualização de mapa sempre ativa
    local_partida = [-23.0838, -47.1336]  # Coordenadas fixas de partida
    if not pedidos_df.empty and 'Latitude' in pedidos_df.columns and 'Longitude' in pedidos_df.columns:
        st.subheader("Mapa dos Pedidos para Roteirizar")
        m = folium.Map(location=[pedidos_df['Latitude'].mean(), pedidos_df['Longitude'].mean()], zoom_start=10)
        for _, row in pedidos_df.iterrows():
            folium.Marker([row['Latitude'], row['Longitude']], popup=row.get('Nome Cliente', '')).add_to(m)
        folium.Marker(local_partida, popup="Local de Partida", icon=folium.Icon(color='red')).add_to(m)
        folium_static(m, width=1200, height=500)
    else:
        st.subheader("Mapa dos Pedidos para Roteirizar")
        m = folium.Map(location=local_partida, zoom_start=10)
        folium.Marker(local_partida, popup="Local de Partida", icon=folium.Icon(color='red')).add_to(m)
        folium_static(m, width=1200, height=500)
        st.info("Sua planilha precisa ter as colunas 'Latitude' e 'Longitude' para exibir os pedidos no mapa.")
    # Botão para iniciar roteirização (placeholder)
    if st.button("Roteirizar Pedidos"):
        st.info("Função de roteirização a ser implementada aqui!")
        # Aqui você pode chamar o algoritmo de roteirização e exibir resultados
