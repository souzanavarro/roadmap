import streamlit as st
import pandas as pd
import os
from geocode import obter_coordenadas_com_fallback
from sklearn.cluster import KMeans
import joblib

def aprender_padroes(historico_path="src/database/historico_roteirizacoes.csv", n_clusters=5):
    df = pd.read_csv(historico_path)
    
    # clustering por localização
    coords = df[["latitude", "longitude"]]
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df["cluster"] = kmeans.fit_predict(coords)

    # salvar modelo de cluster
    joblib.dump(kmeans, "data/modelo_kmeans.pkl")

    # aprender padrão de veículo por cluster
    cluster_veiculo = df.groupby("cluster")["veiculo"].agg(lambda x: x.mode().iloc[0])
    cluster_veiculo.to_csv("src/database/padrao_cluster_veiculo.csv")

    return df, cluster_veiculo

def dashboard_ia():
    st.header(":robot_face: Dashboard IA - Base de Roteirizações")
    st.markdown("""
    <style>
    #dashboard-ia .ia-title {
        font-size: 2.2em;
        font-weight: bold;
        color: #8e24aa;
        margin-bottom: 0.2em;
    }
    #dashboard-ia .ia-box {
        background: linear-gradient(90deg, #f3e5f5 0%, #ce93d8 100%);
        border-radius: 12px;
        padding: 1.5em 2em;
        margin-bottom: 1.5em;
        box-shadow: 0 2px 8px rgba(142,36,170,0.08);
    }
    </style>
    <div id='dashboard-ia'>
      <div class='ia-box'>
          <div class='ia-title'>Base de Roteirizações Inteligente</div>
          <span>Faça upload, visualize e analise roteirizações geradas por IA de forma prática e visual.</span>
      </div>
    """, unsafe_allow_html=True)
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
            st.warning("Sua planilha não possui coordenadas. Como o volume é muito grande, recomendamos obter as coordenadas em lotes menores (ex: 1000 por vez) para não exceder limites das APIs gratuitas. Você pode usar ferramentas como Google Maps, QGIS ou geocodificação em massa para gerar as coordenadas antes do upload.")
            st.info("Se quiser tentar mesmo assim, clique no botão abaixo. O processo pode ser demorado e pode falhar se atingir o limite das APIs.")
            if st.button("Tentar obter coordenadas automaticamente (pode ser lento)"):
                coordenadas_salvas = {}
                coord_db_path = os.path.join("src", "database", "database_coordernadas.csv")
                # Carregar coordenadas já salvas
                if os.path.exists(coord_db_path) and os.path.getsize(coord_db_path) > 0:
                    coord_db = pd.read_csv(coord_db_path)
                    for _, row in coord_db.iterrows():
                        coordenadas_salvas[row['Endereço']] = (row['Latitude'], row['Longitude'])
                def get_coords(row):
                    lat, lon = obter_coordenadas_com_fallback(row['Endereço Completo'], coordenadas_salvas)
                    return pd.Series({'Latitude': lat, 'Longitude': lon})
                ia_df[['Latitude', 'Longitude']] = ia_df.apply(get_coords, axis=1)
                # Salvar coordenadas em database_coordernadas.csv
                coord_df = pd.DataFrame({
                    'Endereço': ia_df['Endereço Completo'],
                    'Latitude': ia_df['Latitude'],
                    'Longitude': ia_df['Longitude']
                })
                coord_df.drop_duplicates(subset=['Endereço'], inplace=True)
                if os.path.exists(coord_db_path):
                    coord_db = pd.read_csv(coord_db_path)
                    coord_df = pd.concat([coord_db, coord_df]).drop_duplicates(subset=['Endereço'], keep='last')
                coord_df.to_csv(coord_db_path, index=False)
                st.success("Coordenadas obtidas (parcialmente ou totalmente) e salvas no banco de coordenadas. Confira a planilha!")
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
    st.markdown("""
    <style>
    #dashboard-ia-mapa .map-box {
        background: linear-gradient(90deg, #f3e5f5 0%, #ce93d8 100%);
        border-radius: 12px;
        padding: 1.5em 2em;
        margin-bottom: 2em;
        box-shadow: 0 2px 8px rgba(142,36,170,0.08);
    }
    #dashboard-ia-mapa .map-title {
        font-size: 1.5em;
        font-weight: bold;
        color: #8e24aa;
        margin-bottom: 0.7em;
    }
    </style>
    <div id='dashboard-ia-mapa'>
      <div class='map-box'>
        <div class='map-title'>Mapa dos Pedidos Roteirizados</div>
    """, unsafe_allow_html=True)
    if not df_map.empty and 'Latitude' in df_map.columns and 'Longitude' in df_map.columns:
        m = folium.Map(location=[df_map['Latitude'].mean(), df_map['Longitude'].mean()], zoom_start=10)
        for _, row in df_map.iterrows():
            folium.Marker([row['Latitude'], row['Longitude']], popup=row.get('Nome Cliente', '')).add_to(m)
        folium.Marker(local_partida, popup="Local de Partida", icon=folium.Icon(color='red')).add_to(m)
        folium_static(m, width=1200, height=500)
    else:
        m = folium.Map(location=local_partida, zoom_start=10)
        folium.Marker(local_partida, popup="Local de Partida", icon=folium.Icon(color='red')).add_to(m)
        folium_static(m, width=1200, height=500)
        st.info("Sua planilha precisa ter as colunas 'Latitude' e 'Longitude' para exibir o mapa.")
    st.markdown("</div></div>", unsafe_allow_html=True)

    st.subheader("🔎 Aprendizado de Padrões de Rotas (MVP)")
    if st.button("Aprender padrões do histórico de roteirizações"):
        try:
            df, cluster_veiculo = aprender_padroes()
            st.success("Padrões aprendidos com sucesso!")
            st.write("Sugestão de veículo por cluster (região):")
            st.dataframe(cluster_veiculo.reset_index())
        except Exception as e:
            st.error(f"Erro ao aprender padrões: {e}")

    st.markdown("""
    <details>
    <summary>Como funciona?</summary>
    <ul>
      <li>O sistema agrupa entregas por localização (KMeans).</li>
      <li>Para cada região (cluster), identifica o veículo mais frequente.</li>
      <li>Esses padrões podem ser usados para sugerir veículos automaticamente em novas roteirizações.</li>
    </ul>
    </details>
    """, unsafe_allow_html=True)
