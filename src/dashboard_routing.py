import streamlit as st
import pandas as pd
import os
from routing import resolver_vrp

def dashboard_routing():
    st.header("Dashboard de Roteirização de Pedidos")

    # Opções de roteirização (radio button)
    st.subheader("Opções de Roteirização")
    tipo_roteirizacao = st.radio(
        "Escolha o tipo de roteirização:",
        options=["VRP (múltiplos veículos)", "TSP (um único veículo)"],
        index=0
    )
    usar_vrp = tipo_roteirizacao.startswith("VRP")
    usar_tsp = tipo_roteirizacao.startswith("TSP")

    # Carregar pedidos e frota
    pedidos_db_path = "src/database/database_pedidos.csv"
    frota_db_path = "src/database/database_frota.csv"

    if not (os.path.exists(pedidos_db_path) and os.path.exists(frota_db_path)):
        st.error("Certifique-se de que os dados de pedidos e frota estão disponíveis.")
        return

    pedidos_df = pd.read_csv(pedidos_db_path)
    frota_df = pd.read_csv(frota_db_path)

    if pedidos_df.empty or frota_df.empty:
        st.error("Os dados de pedidos ou frota estão vazios.")
        return

    # Filtrar pedidos com coordenadas válidas
    pedidos_df = pedidos_df.dropna(subset=['Latitude', 'Longitude'])
    pedidos_df = pedidos_df[pedidos_df['Latitude'].apply(lambda x: pd.notnull(x) and x != '' and x != 0)]
    pedidos_df = pedidos_df[pedidos_df['Longitude'].apply(lambda x: pd.notnull(x) and x != '' and x != 0)]
    if pedidos_df.empty:
        st.error("Nenhum pedido possui coordenadas válidas para roteirização.")
        return

    # Resolver o problema de roteirização conforme a opção escolhida
    rotas = None  # Inicializa rotas
    try:
        if usar_vrp:
            rotas = resolver_vrp(pedidos_df, frota_df)
            st.success("Roteirização VRP concluída com sucesso!")
        elif usar_tsp:
            st.info("Função TSP ainda não implementada neste dashboard.")
        else:
            st.warning("Selecione uma opção de roteirização.")

        # Só mostra filtros e mapas se rotas foi definida
        if rotas:
            veiculos_opcoes = [f"Veículo {i+1} - Placa: {frota_df.iloc[i % len(frota_df)]['Placa']}" for i in range(len(rotas))]
            veiculo_idx = st.selectbox("Selecione o veículo para visualizar a rota:", options=list(range(len(rotas))), format_func=lambda i: veiculos_opcoes[i])

            # Exibir mini planilha do veículo selecionado
            st.subheader(f"Rota para {veiculos_opcoes[veiculo_idx]}")
            pedidos_rota = pedidos_df.iloc[rotas[veiculo_idx]].copy()
            pedidos_rota['Placa'] = frota_df.iloc[veiculo_idx % len(frota_df)]['Placa']
            st.dataframe(pedidos_rota, use_container_width=True)

            # Exibir mapa da rota do veículo selecionado
            from routing import criar_mapa_rotas
            from streamlit_folium import folium_static
            mapa = criar_mapa_rotas(pedidos_rota, rotas=[[i for i in range(len(pedidos_rota))]], partida_coords=(-23.0838, -47.1336))
            folium_static(mapa, width=1600, height=700)

    except Exception as e:
        st.error(f"Erro ao realizar a roteirização: {e}")

def exportar_rotas_para_planilhas(pedidos_df, rotas, pasta_saida='src/database/rotas_exportadas'):
    os.makedirs(pasta_saida, exist_ok=True)
    arquivos_gerados = []
    for veiculo_id, rota in enumerate(rotas):
        pedidos_rota = pedidos_df.iloc[rota]
        total_entregas = pedidos_rota['Qtde. dos Itens'].sum()
        total_peso = pedidos_rota['Peso dos Itens'].sum()
        pedidos_rota = pedidos_rota.copy()
        pedidos_rota['Total Entregas'] = total_entregas
        pedidos_rota['Total Peso'] = total_peso
        nome_arquivo = f'rota_veiculo_{veiculo_id+1}.xlsx'
        caminho_arquivo = os.path.join(pasta_saida, nome_arquivo)
        pedidos_rota.to_excel(caminho_arquivo, index=False)
        arquivos_gerados.append((nome_arquivo, caminho_arquivo))
    return arquivos_gerados
