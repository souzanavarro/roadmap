import streamlit as st
import pandas as pd
import os
from geocode import obter_coordenadas_com_fallback

def dashboard_pedidos():
    st.header(":clipboard: Dashboard de Pedidos")
    st.markdown("""
    <style>
    #dashboard-pedidos .pedidos-title {
        font-size: 2.2em;
        font-weight: bold;
        color: #43a047;
        margin-bottom: 0.2em;
        letter-spacing: 1px;
        text-shadow: 1px 1px 2px #c8e6c9;
    }
    #dashboard-pedidos .pedidos-box {
        background: linear-gradient(90deg, #e8f5e9 0%, #c8e6c9 100%);
        border-radius: 16px;
        padding: 2em 2.5em;
        margin-bottom: 2em;
        box-shadow: 0 4px 16px rgba(67,160,71,0.10);
        border: 1.5px solid #a5d6a7;
        transition: box-shadow 0.3s;
    }
    #dashboard-pedidos .pedidos-box:hover {
        box-shadow: 0 8px 32px rgba(67,160,71,0.18);
    }
    #dashboard-pedidos .pedidos-desc {
        font-size: 1.15em;
        color: #388e3c;
        margin-bottom: 0.5em;
    }
    #dashboard-pedidos .pedidos-help {
        font-size: 1em;
        color: #1b5e20;
        background: #e8f5e9;
        border-radius: 8px;
        padding: 0.7em 1em;
        margin-bottom: 1em;
        border-left: 4px solid #43a047;
    }
    #dashboard-pedidos .loading-coords {
        display: flex;
        align-items: center;
        gap: 1em;
        font-size: 1.2em;
        color: #388e3c;
        font-weight: bold;
        margin-bottom: 1em;
    }
    #dashboard-pedidos .loading-spinner {
        border: 4px solid #c8e6c9;
        border-top: 4px solid #43a047;
        border-radius: 50%;
        width: 32px;
        height: 32px;
        animation: spin 1s linear infinite;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    </style>
    <div id='dashboard-pedidos'>
      <div class='pedidos-box'>
          <div class='pedidos-title'>Gestão de Pedidos</div>
          <div class='pedidos-desc'>Gerencie, edite e visualize seus pedidos de forma prática, visual e moderna.</div>
          <div class='pedidos-help'>Dica: Faça upload da planilha de pedidos, confira as coordenadas e avance para a roteirização!</div>
      </div>
    """, unsafe_allow_html=True)
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
        # Obter coordenadas se não existirem
        if 'Latitude' not in pedidos_df.columns or 'Longitude' not in pedidos_df.columns or pedidos_df['Latitude'].isnull().any() or pedidos_df['Longitude'].isnull().any():
            st.markdown("""
            <div class='loading-coords'>
                <div class='loading-spinner'></div>
                Obtendo coordenadas para os endereços...
            </div>
            """, unsafe_allow_html=True)
            coordenadas_salvas = {}
            # Carregar coordenadas já salvas
            coord_db_path = os.path.join("src", "database", "database_coordernadas.csv")
            if os.path.exists(coord_db_path) and os.path.getsize(coord_db_path) > 0:
                coord_db = pd.read_csv(coord_db_path)
                for _, row in coord_db.iterrows():
                    coordenadas_salvas[row['Endereço']] = (row['Latitude'], row['Longitude'])
            api_key = "6f522c67add14152926990afbe127384"
            # Priorizar OpenCage e usar Nominatim como fallback
            def get_coords(row):
                endereco = f"{row['Endereço de Entrega']}, {row['Bairro de Entrega']}, {row['Cidade de Entrega']}"
                if endereco in coordenadas_salvas:
                    lat, lon = coordenadas_salvas[endereco]
                else:
                    try:
                        api_keys = [
                            "5161dbd006cf4c43a7f7dd789ee1a3da",
                            "6f522c67add14152926990afbe127384",
                            "6c2d02cafb2e4b49aa3485a62262e54b"
                        ]
                        lat, lon = obter_coordenadas_com_fallback(endereco, coordenadas_salvas)
                        # Trata caso a API não retorne resultado
                        if lat is None or lon is None:
                            st.warning(f"Não foi possível obter coordenadas para: {endereco}. Deixando valores como nulos.")
                            lat, lon = None, None
                    except Exception as e:
                        st.warning(f"Erro ao tentar obter as coordenadas para '{endereco}': {e}. Deixando valores como nulos.")
                        lat, lon = None, None
                return pd.Series({'Latitude': lat, 'Longitude': lon})
            coords = pedidos_df.apply(get_coords, axis=1)
            pedidos_df['Latitude'] = coords['Latitude']
            pedidos_df['Longitude'] = coords['Longitude']
            # Salvar coordenadas em database_coordernadas.csv
            # Agora também salva a Região
            coord_df = pd.DataFrame({
                'Endereço': pedidos_df['Endereço de Entrega'] + ', ' + pedidos_df['Bairro de Entrega'] + ', ' + pedidos_df['Cidade de Entrega'],
                'Latitude': pedidos_df['Latitude'],
                'Longitude': pedidos_df['Longitude'],
                'Região': pedidos_df.get('Região', None)  # Salva a Região se já existir
            })
            coord_df.drop_duplicates(subset=['Endereço'], inplace=True)
            if os.path.exists(coord_db_path):
                coord_db = pd.read_csv(coord_db_path)
                # Se o banco antigo não tem a coluna Região, adiciona
                if 'Região' not in coord_db.columns:
                    coord_db['Região'] = None
                # Concatena e mantém a última ocorrência
                coord_df = pd.concat([coord_db, coord_df]).drop_duplicates(subset=['Endereço'], keep='last')
            coord_df.to_csv(coord_db_path, index=False)
            st.success("Coordenadas obtidas com sucesso!")

            # Preencher regiões reais dos pedidos a partir das coordenadas
            from geocode import preencher_regioes_pedidos
            st.info('Preenchendo regiões reais dos pedidos a partir das coordenadas (pode demorar alguns minutos na primeira vez)...')
            pedidos_df = preencher_regioes_pedidos(pedidos_df)
            st.success('Regiões preenchidas automaticamente!')
            st.write(f"[LOG] Regiões detectadas: {pedidos_df['Região'].unique()}")

            # Atualiza o arquivo database_coordernadas.csv com as regiões preenchidas
            coord_df = pd.DataFrame({
                'Endereço': pedidos_df['Endereço de Entrega'] + ', ' + pedidos_df['Bairro de Entrega'] + ', ' + pedidos_df['Cidade de Entrega'],
                'Latitude': pedidos_df['Latitude'],
                'Longitude': pedidos_df['Longitude'],
                'Região': pedidos_df['Região']
            })
            coord_df.drop_duplicates(subset=['Endereço'], inplace=True)
            if os.path.exists(coord_db_path):
                coord_db = pd.read_csv(coord_db_path)
                if 'Região' not in coord_db.columns:
                    coord_db['Região'] = None
                # Atualiza apenas as regiões dos endereços presentes em pedidos_df
                coord_db.set_index('Endereço', inplace=True)
                coord_df.set_index('Endereço', inplace=True)
                coord_db.update(coord_df)
                # Adiciona novos endereços, se houver
                coord_db = pd.concat([coord_db, coord_df[~coord_df.index.isin(coord_db.index)]]).reset_index()
            else:
                coord_db = coord_df.reset_index()
            coord_db.to_csv(coord_db_path, index=False)

            st.dataframe(pedidos_df)
        st.dataframe(pedidos_df)
        # Botão para limpar e salvar novos dados
        if st.button("Limpar e Salvar Novos Pedidos", type="primary"):
            # Limpa o database_pedidos.csv
            pedidos_db_path = os.path.join("src", "database", "database_pedidos.csv")
            pedidos_df.to_csv(pedidos_db_path, index=False)
            # Limpa o historico_roteirizacoes.csv
            historico_path = os.path.join("src", "database", "historico_roteirizacoes.csv")
            if os.path.exists(historico_path):
                with open(historico_path, "w") as f:
                    f.write("")
            st.success("Pedidos atualizados e histórico de roteirizações limpo!")
        # Salvar no database local na pasta src/database
        os.makedirs(os.path.join("src", "database"), exist_ok=True)
        pedidos_db_path = os.path.join("src", "database", "database_pedidos.csv")
        pedidos_df.to_csv(pedidos_db_path, index=False)
        st.success(f"Pedidos salvos no banco de dados local: {pedidos_db_path}")
        df_map = pedidos_df
        # Agrupamento por região (clusters)
        st.subheader("Agrupar pedidos por região (clusters)")
        n_clusters = st.number_input('Número de regiões (clusters)', min_value=1, max_value=20, value=3, step=1)
        if st.button('Agrupar por Região'):
            from routing import agrupar_por_regiao
            pedidos_df = agrupar_por_regiao(pedidos_df, n_clusters)
            st.success(f'Pedidos agrupados em {n_clusters} regiões!')
            st.dataframe(pedidos_df)
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
    # Remover o visual customizado do mapa dos pedidos
    if not df_map.empty and 'Latitude' in df_map.columns and 'Longitude' in df_map.columns:
        df_map_valid = df_map.dropna(subset=['Latitude', 'Longitude'])
        m = folium.Map(location=[df_map_valid['Latitude'].mean() if not df_map_valid.empty else local_partida[0],
                                 df_map_valid['Longitude'].mean() if not df_map_valid.empty else local_partida[1]], zoom_start=10)
        for _, row in df_map_valid.iterrows():
            folium.Marker([row['Latitude'], row['Longitude']], popup=row.get('Nome Cliente', '')).add_to(m)
        folium.Marker(local_partida, popup="Local de Partida", icon=folium.Icon(color='red')).add_to(m)
        folium_static(m, width=1200, height=500)
    else:
        m = folium.Map(location=local_partida, zoom_start=10)
        folium.Marker(local_partida, popup="Local de Partida", icon=folium.Icon(color='red')).add_to(m)
        folium_static(m, width=1200, height=500)
        st.info("Sua planilha precisa ter as colunas 'Latitude' e 'Longitude' para exibir os pedidos no mapa.")
    st.markdown("</div></div>", unsafe_allow_html=True)
