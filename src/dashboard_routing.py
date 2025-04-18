import streamlit as st
import pandas as pd
import os
from routing import resolver_vrp, alocacao_prioridade_capacidade_regiao

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
    st.write("[LOG] Iniciando dashboard de roteirização...")
    tipo_roteirizacao = st.radio(
        "Escolha o tipo de roteirização:",
        options=["VRP (múltiplos veículos)", "TSP (um único veículo)"],
        index=0
    )
    st.write(f"[LOG] Tipo de roteirização selecionado: {tipo_roteirizacao}")
    usar_vrp = tipo_roteirizacao.startswith("VRP")
    usar_tsp = tipo_roteirizacao.startswith("TSP")

    # Carregar pedidos e frota
    pedidos_db_path = "src/database/database_pedidos.csv"
    frota_db_path = "src/database/database_frota.csv"

    st.write(f"[LOG] Verificando existência dos arquivos: {pedidos_db_path}, {frota_db_path}")
    if not (os.path.exists(pedidos_db_path) and os.path.exists(frota_db_path)):
        st.error("Certifique-se de que os dados de pedidos e frota estão disponíveis.")
        print("[LOG] Arquivos de pedidos ou frota não encontrados!")
        return

    pedidos_df = pd.read_csv(pedidos_db_path)
    frota_df = pd.read_csv(frota_db_path)
    st.write(f"[LOG] Pedidos carregados: {len(pedidos_df)} | Frota carregada: {len(frota_df)}")

    # Preencher regiões reais a partir das coordenadas antes de qualquer agrupamento
    from geocode import preencher_regioes_pedidos
    if 'Região' not in pedidos_df.columns or pedidos_df['Região'].isnull().all():
        st.info('Preenchendo regiões reais dos pedidos a partir das coordenadas (pode demorar alguns minutos na primeira vez)...')
        pedidos_df = preencher_regioes_pedidos(pedidos_df)
        st.success('Regiões preenchidas automaticamente!')
        st.write(f"[LOG] Regiões detectadas: {pedidos_df['Região'].unique()}")

    # Atualizar total_regioes para refletir as regiões reais
    total_regioes = pedidos_df['Região'].nunique()
    st.info(f'Total de regiões detectadas: {total_regioes}')

    # Opções de restrições e parâmetros avançados
    with st.expander('Restrições e Parâmetros Avançados'):
        modo_capacidade = st.radio('Como considerar a capacidade dos veículos?', [
            'Capacidade individual da frota (%)',
            'Capacidade máxima fixa (kg)'
        ], index=0)
        if modo_capacidade == 'Capacidade máxima fixa (kg)':
            capacidade_total_frota = int(frota_df['Capac. Kg'].sum()) if 'Capac. Kg' in frota_df.columns else 0
            st.info(f'Capacidade total disponível na frota: {capacidade_total_frota} kg')
            capacidade_max = st.number_input('Capacidade máxima por veículo (kg)', min_value=1, value=capacidade_total_frota)
            percentual_utilizacao = 100
        else:
            percentual_utilizacao = st.slider('Percentual de utilização da capacidade de cada caminhão (%)', min_value=10, max_value=100, value=100, step=5)
            capacidade_max = None
        usar_janela_tempo = st.checkbox('Usar janelas de tempo para entregas?')
        if usar_janela_tempo:
            janela_inicio = st.time_input('Início da janela de entrega', value=None)
            janela_fim = st.time_input('Fim da janela de entrega', value=None)
        tipo_otimizacao = st.selectbox('Tipo de otimização', ['Menor distância', 'Menor tempo'])
        prioridade_alocacao = st.selectbox('Prioridade de alocação', ['Capacidade', 'Região', 'Capacidade + Região', 'Tipo de carga'])
        n_clusters = 3  # valor padrão
        regioes_por_veiculo = 1
        if 'Região' in prioridade_alocacao:
            # Determinar total de regiões reais
            total_regioes = pedidos_df['Região'].nunique()
            st.info(f'Total de regiões detectadas: {total_regioes}')
            percentual_regioes = st.slider('Percentual de regiões (clusters)', min_value=5, max_value=100, value=10, step=5, help='Percentual de clusters em relação ao total de regiões distintas nos pedidos')
            n_clusters = max(1, int(total_regioes * percentual_regioes / 100))
            # CORREÇÃO: Slider só aparece se n_clusters > 1
            if n_clusters > 1:
                regioes_por_veiculo = st.slider('Máximo de regiões por veículo', min_value=1, max_value=n_clusters, value=1)
            else:
                regioes_por_veiculo = 1

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

    rotas = None  # Inicializa rotas
    if st.button("Roteirizar Pedidos", type="primary"):
        try:
            st.write(f"[LOG] Iniciando roteirização com {len(pedidos_df)} pedidos e {len(frota_df)} veículos.")
            # Antes de roteirizar, garantir que as regiões estão separadas por coordenadas
            if 'Região' not in pedidos_df.columns or pedidos_df['Região'].nunique() < n_clusters:
                from routing import agrupar_por_regiao
                st.write(f"[LOG] Agrupando pedidos em {n_clusters} regiões...")
                pedidos_df = agrupar_por_regiao(pedidos_df, n_clusters)
                st.write(f"[LOG] Regiões após agrupamento: {pedidos_df['Regiao'].unique()}")

            if usar_vrp:
                st.write(f"[LOG] Chamando algoritmo VRP com prioridade: {prioridade_alocacao}")
                if prioridade_alocacao == 'Capacidade + Região':
                    pedidos_alocados_df = alocacao_prioridade_capacidade_regiao(pedidos_df, frota_df, n_clusters=n_clusters)
                    st.success("Alocação por Capacidade + Região concluída! Veja a tabela abaixo.")
                    st.dataframe(pedidos_alocados_df, use_container_width=True)
                    print("[LOG] Alocação por Capacidade + Região finalizada.")
                    return
                elif prioridade_alocacao == 'Região':
                    from routing import alocacao_regioes_por_veiculo
                    pedidos_alocados_df = alocacao_regioes_por_veiculo(pedidos_df, frota_df, n_clusters=n_clusters, regioes_por_veiculo=regioes_por_veiculo)
                    st.success("Alocação por Região (com limite de regiões por veículo) concluída! Veja a tabela abaixo.")
                    st.dataframe(pedidos_alocados_df, use_container_width=True)
                    print("[LOG] Alocação por Região finalizada.")
                    return
                else:
                    if modo_capacidade == 'Capacidade máxima fixa (kg)':
                        st.write(f"[LOG] VRP com capacidade máxima fixa: {capacidade_max}")
                        rotas = resolver_vrp(pedidos_df, frota_df, capacidade_max=capacidade_max)
                    else:
                        st.write(f"[LOG] VRP com percentual de utilização: {percentual_utilizacao}%")
                        rotas = resolver_vrp(pedidos_df, frota_df, percentual_utilizacao=percentual_utilizacao)
                    st.success("Roteirização VRP concluída e salva no histórico!")
                    st.write(f"[LOG] Rotas geradas: {rotas}")
            elif usar_tsp:
                from routing import tsp_nearest_neighbor
                partida_coords = (-23.0838, -47.1336)
                st.write(f"[LOG] Chamando algoritmo TSP para {len(pedidos_df)} pedidos.")
                rota_tsp = tsp_nearest_neighbor(pedidos_df, partida_coords)
                rotas = [rota_tsp]
                st.success("Roteirização TSP concluída e salva no histórico!")
                st.write(f"[LOG] Rota TSP gerada: {rota_tsp}")
            else:
                st.warning("Selecione uma opção de roteirização.")
                print("[LOG] Nenhuma opção de roteirização selecionada.")

            # Garantir que cada pedido seja alocado em apenas um veículo
            pedidos_alocados = set()
            rotas_unicas = []
            for rota in rotas:
                rota_unica = [idx for idx in rota if idx not in pedidos_alocados]
                pedidos_alocados.update(rota_unica)
                rotas_unicas.append(rota_unica)
            rotas = rotas_unicas
            st.write(f"[LOG] Rotas finais (sem duplicidade): {rotas}")

            # Salvar histórico de roteirizações
            from datetime import datetime
            historico_path = os.path.join("src", "database", "historico_roteirizacoes.csv")
            historico = []
            data_roteirizacao = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for veiculo_id, rota in enumerate(rotas):
                if len(rota) == 0:
                    continue
                pedidos_rota = pedidos_df.iloc[rota].copy()
                if usar_tsp:
                    pedidos_rota['Placa'] = 'TSP'
                else:
                    pedidos_rota['Placa'] = frota_df.iloc[veiculo_id % len(frota_df)]['Placa']
                pedidos_rota['Veiculo'] = veiculo_id + 1
                pedidos_rota['Data Roteirizacao'] = data_roteirizacao
                historico.append(pedidos_rota)
            if historico:
                historico_df = pd.concat(historico)
                historico_df.to_csv(historico_path, index=False)
                st.write(f"[LOG] Histórico salvo em {historico_path} com {len(historico_df)} registros.")
        except Exception as e:
            tratar_erro("Erro ao realizar a roteirização. Verifique os dados e tente novamente.", e)
            st.write(f"[LOG] Erro na roteirização: {e}")
        return  # Não mostrar nada até o usuário clicar novamente

    # --- Distribuição da Carga por Veículo ---
    st.subheader('Distribuição da Carga por Veículo')
    if 'Peso Total' not in pedidos_df.columns and 'Peso dos Itens' in pedidos_df.columns and 'Qtde. dos Itens' in pedidos_df.columns:
        pedidos_df['Peso Total'] = pedidos_df['Peso dos Itens'] * pedidos_df['Qtde. dos Itens']
    if 'Volume Total' not in pedidos_df.columns and 'Volume Unitário' in pedidos_df.columns and 'Qtde. dos Itens' in pedidos_df.columns:
        pedidos_df['Volume Total'] = pedidos_df['Volume Unitário'] * pedidos_df['Qtde. dos Itens']
    # Verificar disponibilidade dos veículos
    if 'Disponível' in frota_df.columns:
        frota_disponivel = frota_df[frota_df['Disponível'].str.lower() == 'sim']
    else:
        frota_disponivel = frota_df.copy()
    if frota_disponivel.empty:
        st.error('Nenhum veículo disponível para roteirização!')
        return
    # Agrupar pedidos por cluster/região
    if 'Regiao' in pedidos_df.columns:
        pedidos_df['Cluster'] = pedidos_df['Regiao']
    elif 'Região' in pedidos_df.columns:
        pedidos_df['Cluster'] = pedidos_df['Região']
    else:
        pedidos_df['Cluster'] = 0
    clusters = pedidos_df['Cluster'].unique()
    carga_veiculos = []
    veiculo_idx = 0
    for cluster in clusters:
        pedidos_cluster = pedidos_df[pedidos_df['Cluster'] == cluster]
        peso_total = pedidos_cluster['Peso Total'].sum() if 'Peso Total' in pedidos_cluster.columns else 0
        volume_total = pedidos_cluster['Volume Total'].sum() if 'Volume Total' in pedidos_cluster.columns else 0
        pedidos_restantes = pedidos_cluster.copy()
        while not pedidos_restantes.empty and veiculo_idx < len(frota_disponivel):
            veiculo = frota_disponivel.iloc[veiculo_idx]
            capacidade_kg = veiculo['Capac. Kg'] if 'Capac. Kg' in veiculo else 0
            capacidade_vol = veiculo['Capac. Cx'] if 'Capac. Cx' in veiculo else None
            peso_usado = 0
            volume_usado = 0
            pedidos_alocados_idx = []
            for idx, pedido in pedidos_restantes.iterrows():
                peso_pedido = pedido['Peso Total'] if 'Peso Total' in pedido else 0
                volume_pedido = pedido['Volume Total'] if 'Volume Total' in pedido else 0
                if (peso_usado + peso_pedido <= capacidade_kg) and (capacidade_vol is None or volume_usado + volume_pedido <= capacidade_vol):
                    peso_usado += peso_pedido
                    volume_usado += volume_pedido
                    pedidos_alocados_idx.append(idx)
            percentual_peso = round(100 * peso_usado / capacidade_kg, 1) if capacidade_kg else 0
            percentual_vol = round(100 * volume_usado / capacidade_vol, 1) if capacidade_vol else None
            carga_veiculos.append({
                'Veículo': veiculo['Placa'] if 'Placa' in veiculo else veiculo_idx+1,
                'Cluster/Região': cluster,
                'Peso Alocado (kg)': peso_usado,
                'Capacidade (kg)': capacidade_kg,
                'Aproveitamento Peso (%)': percentual_peso,
                'Volume Alocado': volume_usado,
                'Capacidade Volume': capacidade_vol,
                'Aproveitamento Volume (%)': percentual_vol
            })
            pedidos_restantes = pedidos_restantes.drop(pedidos_alocados_idx)
            veiculo_idx += 1
        if not pedidos_restantes.empty:
            st.warning(f"Cluster/Região {cluster}: Excesso de carga! {len(pedidos_restantes)} pedidos não alocados.")
    carga_veiculos_df = pd.DataFrame(carga_veiculos)
    st.dataframe(carga_veiculos_df, use_container_width=True)

    # --- Otimização da Rota por Veículo ---
    st.subheader('Otimização da Rota (Ordem de Entrega)')
    criterio_otimizacao = st.selectbox('Critério de otimização da rota', ['Menor distância', 'Menor tempo'])
    usar_janelas = 'Janela Inicial' in pedidos_df.columns and 'Janela Final' in pedidos_df.columns
    ponto_partida = st.text_input('Endereço ou coordenada de partida do veículo', value='-23.0838, -47.1336')
    if st.button('Otimizar Rotas de Entrega por Veículo'):
        from routing import get_osrm_distance_matrix, resolver_vrp
        import numpy as np
        resultados_rotas = []
        for veiculo in carga_veiculos_df['Veículo'].unique():
            pedidos_veic = pedidos_df[pedidos_df['Placa'] == veiculo] if 'Placa' in pedidos_df.columns else pedidos_df.iloc[[]]
            if pedidos_veic.empty:
                continue
            coords = [(row['Latitude'], row['Longitude']) for _, row in pedidos_veic.iterrows()]
            # Adiciona ponto de partida
            try:
                lat0, lon0 = map(float, ponto_partida.split(','))
                coords = [(lat0, lon0)] + coords
            except:
                coords = coords
            matriz = None
            if criterio_otimizacao == 'Menor tempo':
                matriz = get_osrm_distance_matrix(coords)
            # Janelas de entrega
            janelas = None
            if usar_janelas:
                janelas = list(zip(pedidos_veic['Janela Inicial'], pedidos_veic['Janela Final']))
                # Adiciona janela ampla para o ponto de partida
                janelas = [(0, 1440)] + janelas
            rotas = resolver_vrp(
                pedidos_veic.reset_index(drop=True),
                pd.DataFrame([{'Placa': veiculo, 'Capac. Kg': carga_veiculos_df[carga_veiculos_df['Veículo']==veiculo]['Capacidade (kg)'].iloc[0]}]),
                matriz_distancias=matriz,
            )
            resultados_rotas.append({'Veículo': veiculo, 'Rota': rotas[0] if rotas else [], 'Pedidos': pedidos_veic})
        for res in resultados_rotas:
            st.markdown(f"**Veículo:** {res['Veículo']}")
            if res['Rota']:
                pedidos_ordenados = res['Pedidos'].iloc[res['Rota']]
                st.dataframe(pedidos_ordenados)
            else:
                st.warning('Não foi possível otimizar a rota para este veículo.')

    # --- Geração e Exportação das Rotas ---
    st.subheader('Geração e Exportação das Rotas')
    if 'resultados_rotas' in locals() and resultados_rotas:
        from routing import criar_mapa_rotas, exportar_rotas_para_planilhas
        from streamlit_folium import folium_static
        for res in resultados_rotas:
            st.markdown(f"### Veículo: {res['Veículo']}")
            if res['Rota']:
                pedidos_ordenados = res['Pedidos'].iloc[res['Rota']]
                # Visualização no mapa
                mapa = criar_mapa_rotas(pedidos_ordenados.reset_index(drop=True), rotas=[[i for i in range(len(pedidos_ordenados))]], partida_coords=(-23.0838, -47.1336))
                folium_static(mapa, width=1000, height=400)
                # Exportar rota Excel
                pasta_saida = os.path.join('src', 'database', 'rotas_exportadas')
                arquivos = exportar_rotas_para_planilhas(pedidos_ordenados, [[i for i in range(len(pedidos_ordenados))]], pasta_saida=pasta_saida)
                for nome_arquivo, caminho_arquivo in arquivos:
                    st.success(f"Rota exportada: {nome_arquivo}")
                    st.download_button(f"Baixar {nome_arquivo}", data=open(caminho_arquivo, 'rb').read(), file_name=nome_arquivo)
                # Salvar no histórico
                historico_path = os.path.join("src", "database", "historico_roteirizacoes.csv")
                from datetime import datetime
                pedidos_ordenados['Data Roteirizacao'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                if os.path.exists(historico_path):
                    historico_df = pd.read_csv(historico_path)
                    historico_df = pd.concat([historico_df, pedidos_ordenados], ignore_index=True)
                else:
                    historico_df = pedidos_ordenados
                historico_df.to_csv(historico_path, index=False)
                st.info(f"Rota salva no histórico para aprendizado.")
            else:
                st.warning('Não foi possível gerar/exportar a rota para este veículo.')

    # --- Aprendizado de Padrões de Alocação (IA) ---
    st.subheader('Sugestão Inteligente de Veículo por Região (IA)')
    historico_path = os.path.join("src", "database", "historico_roteirizacoes.csv")
    sugestoes_veiculo = {}
    if os.path.exists(historico_path):
        historico_df = pd.read_csv(historico_path)
        if 'Regiao' in historico_df.columns and 'Placa' in historico_df.columns:
            # Aprender: modelo simples de predição (maior frequência)
            freq = historico_df.groupby(['Regiao', 'Placa']).size().reset_index(name='freq')
            sugestoes_veiculo = freq.sort_values('freq', ascending=False).groupby('Regiao').first()['Placa'].to_dict()
            st.info('Sugestão automática: para cada região, o veículo mais frequente será sugerido na alocação.')
            st.dataframe(pd.DataFrame(list(sugestoes_veiculo.items()), columns=['Região', 'Veículo Sugerido']))
    # Aplicar sugestão no roteirizador
    if st.button('Aplicar Sugestão de Veículo por Região'):
        if sugestoes_veiculo:
            if 'Regiao' in pedidos_df.columns:
                pedidos_df['Placa Sugerida'] = pedidos_df['Regiao'].map(sugestoes_veiculo)
                st.success('Sugestão aplicada! Veja a coluna "Placa Sugerida" na tabela de pedidos.')
                st.dataframe(pedidos_df)
            else:
                st.warning('Pedidos não possuem coluna de região para aplicar sugestão.')
        else:
            st.warning('Ainda não há histórico suficiente para sugerir veículos por região.')

    # Após roteirizar, só mostrar o histórico
    historico_path = os.path.join("src", "database", "historico_roteirizacoes.csv")
    if not os.path.exists(historico_path) or os.path.getsize(historico_path) == 0:
        st.info("Nenhuma roteirização foi realizada ainda. Clique em 'Roteirizar Pedidos' para começar.")
        return
    historico_df = pd.read_csv(historico_path)
    if historico_df.empty:
        st.info("Nenhuma roteirização foi realizada ainda. Clique em 'Roteirizar Pedidos' para começar.")
        return
    # Mostrar apenas o que está no histórico
    historico_df['Placa'] = historico_df['Placa'].astype(str)
    veiculos_unicos = historico_df.drop_duplicates(subset=['Veiculo', 'Placa'])[['Veiculo', 'Placa']]
    veiculos_opcoes = [f"Veículo {v} - Placa: {p}" for v, p in zip(veiculos_unicos['Veiculo'], veiculos_unicos['Placa'])]
    veiculo_idx_map = {opcao: v for opcao, v in zip(veiculos_opcoes, veiculos_unicos['Veiculo'])}
    veiculo_selecionado = st.selectbox(
        "Selecione o veículo para visualizar a rota:",
        options=veiculos_opcoes
    )
    veiculo_idx = veiculo_idx_map[veiculo_selecionado]
    pedidos_rota = historico_df[historico_df['Veiculo'] == veiculo_idx].copy()
    st.subheader(f"Rota para {veiculo_selecionado}")
    if pedidos_rota.empty:
        st.info("Este veículo não possui pedidos alocados.")
    else:
        st.write(f"### Veículo: {pedidos_rota['Placa'].iloc[0]} | Total de Pedidos: {len(pedidos_rota)}")
        st.dataframe(pedidos_rota, use_container_width=True)
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
