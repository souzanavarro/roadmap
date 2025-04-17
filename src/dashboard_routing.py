import streamlit as st
import pandas as pd
import os
from routing import resolver_vrp

def tratar_erro(msg, exception=None):
    import streamlit as st
    import datetime
    st.error(msg)
    if exception:
        log_path = "src/database/log_erros.txt"
        with open(log_path, "a") as f:
            f.write(f"{datetime.datetime.now()} - {msg} - {str(exception)}\n")

def dashboard_routing():
    st.header(":bar_chart: Dashboard de Roteirização")
    st.markdown("""
    <style>
    #dashboard-routing .routing-title {
        font-size: 2.2em;
        font-weight: bold;
        color: #ff9800;
        margin-bottom: 0.2em;
    }
    #dashboard-routing .routing-box {
        background: linear-gradient(90deg, #fff3e0 0%, #ffe0b2 100%);
        border-radius: 12px;
        padding: 1.5em 2em;
        margin-bottom: 1.5em;
        box-shadow: 0 2px 8px rgba(255,152,0,0.08);
    }
    </style>
    <div id='dashboard-routing'>
      <div class='routing-box'>
          <div class='routing-title'>Painel de Roteirização</div>
          <span>Visualize, analise e execute a roteirização de pedidos de forma eficiente e visual.</span>
      </div>
    """, unsafe_allow_html=True)

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

    if os.path.exists(frota_db_path):
        frota_df = pd.read_csv(frota_db_path)
        if 'Disponível' in frota_df.columns:
            frota_disponivel = frota_df[frota_df['Disponível'].str.lower() == 'sim']
            total_disponivel = len(frota_disponivel)
            st.info(f"🚚 Veículos disponíveis para roteirização: {total_disponivel}")
        else:
            st.info(f"🚚 Veículos cadastrados: {len(frota_df)}")

    # Filtrar pedidos com coordenadas válidas
    pedidos_df = pedidos_df.dropna(subset=['Latitude', 'Longitude'])
    pedidos_df = pedidos_df[pedidos_df['Latitude'].apply(lambda x: pd.notnull(x) and x != '' and x != 0)]
    pedidos_df = pedidos_df[pedidos_df['Longitude'].apply(lambda x: pd.notnull(x) and x != '' and x != 0)]
    if pedidos_df.empty:
        st.error("Nenhum pedido possui coordenadas válidas para roteirização.")
        return

    # Remover pedidos duplicados por cliente (mantendo o maior peso)
    if 'Cód. Cliente' in pedidos_df.columns and 'Peso dos Itens' in pedidos_df.columns:
        pedidos_df = pedidos_df.sort_values('Peso dos Itens', ascending=False)
        pedidos_df = pedidos_df.drop_duplicates(subset=['Cód. Cliente'], keep='first')

    # Botão para executar a roteirização
    if st.button("Roteirizar Pedidos", type="primary"):
        try:
            if usar_vrp:
                rotas = resolver_vrp(pedidos_df, frota_df)
                st.success("Roteirização VRP concluída e salva no histórico!")
            elif usar_tsp:
                st.info("Função TSP ainda não implementada neste dashboard.")
            else:
                st.warning("Selecione uma opção de roteirização.")

            # Garantir que cada pedido seja alocado em apenas um veículo
            pedidos_alocados = set()
            rotas_unicas = []
            for rota in rotas:
                rota_unica = [idx for idx in rota if idx not in pedidos_alocados]
                pedidos_alocados.update(rota_unica)
                rotas_unicas.append(rota_unica)
            rotas = rotas_unicas

            # Salvar histórico de roteirizações
            from datetime import datetime
            historico_path = os.path.join("src", "database", "historico_roteirizacoes.csv")
            historico = []
            data_roteirizacao = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for veiculo_id, rota in enumerate(rotas):
                if len(rota) == 0:
                    continue
                pedidos_rota = pedidos_df.iloc[rota].copy()
                pedidos_rota['Placa'] = frota_df.iloc[veiculo_id % len(frota_df)]['Placa']
                pedidos_rota['Veiculo'] = veiculo_id + 1
                pedidos_rota['Data Roteirizacao'] = data_roteirizacao
                historico.append(pedidos_rota)
            if historico:
                historico_df = pd.concat(historico)
                historico_df.to_csv(historico_path, index=False)
        except Exception as e:
            tratar_erro("Erro ao realizar a roteirização. Verifique os dados e tente novamente.", e)

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

        # Garantir que cada pedido seja alocado em apenas um veículo
        pedidos_alocados = set()
        rotas_unicas = []
        for rota in rotas:
            rota_unica = [idx for idx in rota if idx not in pedidos_alocados]
            pedidos_alocados.update(rota_unica)
            rotas_unicas.append(rota_unica)
        rotas = rotas_unicas

        # Só mostra filtros e mapas se rotas foi definida
        if rotas:
            # Filtrar apenas veículos com pedidos
            veiculos_com_pedidos = [i for i, rota in enumerate(rotas) if len(rota) > 0]
            if not veiculos_com_pedidos:
                st.warning("Nenhum veículo possui pedidos alocados nesta roteirização.")
                return
            veiculos_opcoes = [f"Veículo {i+1} - Placa: {frota_df.iloc[i % len(frota_df)]['Placa']}" for i in veiculos_com_pedidos]
            veiculo_idx_map = {opcao: i for opcao, i in zip(veiculos_opcoes, veiculos_com_pedidos)}
            veiculo_selecionado = st.selectbox(
                "Selecione o veículo para visualizar a rota:",
                options=veiculos_opcoes
            )
            veiculo_idx = veiculo_idx_map[veiculo_selecionado]

            # Exibir mini planilha do veículo selecionado
            st.subheader(f"Rota para {veiculos_opcoes[veiculo_idx]}")
            if len(rotas[veiculo_idx]) == 0:
                st.info("Este veículo não possui pedidos alocados.")
            else:
                pedidos_rota = pedidos_df.iloc[rotas[veiculo_idx]].copy()
                pedidos_rota['Placa'] = frota_df.iloc[veiculo_idx % len(frota_df)]['Placa']
                st.dataframe(pedidos_rota, use_container_width=True)

                # Exibir mapa da rota do veículo selecionado
                from routing import criar_mapa_rotas
                from streamlit_folium import folium_static
                st.markdown("""
                <style>
                #dashboard-routing-mapa .map-box {
                    background: linear-gradient(90deg, #fff3e0 0%, #ffe0b2 100%);
                    border-radius: 12px;
                    padding: 1.5em 2em;
                    margin-bottom: 2em;
                    box-shadow: 0 2px 8px rgba(255,152,0,0.08);
                }
                #dashboard-routing-mapa .map-title {
                    font-size: 1.5em;
                    font-weight: bold;
                    color: #ff9800;
                    margin-bottom: 0.7em;
                }
                </style>
                <div id='dashboard-routing-mapa'>
                  <div class='map-box'>
                    <div class='map-title'>Mapa da Roteirização</div>
                """, unsafe_allow_html=True)
                mapa = criar_mapa_rotas(pedidos_rota, rotas=[[i for i in range(len(pedidos_rota))]], partida_coords=(-23.0838, -47.1336))
                folium_static(mapa, width=1200, height=500)
                st.markdown("</div></div>", unsafe_allow_html=True)

            # Salvar histórico de roteirizações
            from datetime import datetime
            historico_path = os.path.join("src", "database", "historico_roteirizacoes.csv")
            historico = []
            data_roteirizacao = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for veiculo_id, rota in enumerate(rotas):
                if len(rota) == 0:
                    continue
                pedidos_rota = pedidos_df.iloc[rota].copy()
                pedidos_rota['Placa'] = frota_df.iloc[veiculo_id % len(frota_df)]['Placa']
                pedidos_rota['Veiculo'] = veiculo_id + 1
                pedidos_rota['Data Roteirizacao'] = data_roteirizacao
                historico.append(pedidos_rota)
            if historico:
                historico_df = pd.concat(historico)
                if os.path.exists(historico_path):
                    historico_antigo = pd.read_csv(historico_path)
                    historico_df = pd.concat([historico_antigo, historico_df], ignore_index=True)
                historico_df.to_csv(historico_path, index=False)
    except Exception as e:
        tratar_erro("Erro ao realizar a roteirização. Verifique os dados e tente novamente.", e)

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
