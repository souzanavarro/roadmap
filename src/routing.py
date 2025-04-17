import random
import itertools
import networkx as nx
from geopy.distance import geodesic
from sklearn.cluster import KMeans
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import pandas as pd
import folium
import os
from itertools import permutations

def calcular_distancia(coords_1, coords_2):
    if coords_1 and coords_2:
        return geodesic(coords_1, coords_2).meters
    else:
        return None

def criar_grafo_tsp(pedidos_df, endereco_partida, endereco_partida_coords):
    G = nx.Graph()
    enderecos = pedidos_df['Endereço Completo'].unique()
    G.add_node(endereco_partida, pos=endereco_partida_coords)
    for endereco in enderecos:
        coords = (pedidos_df.loc[pedidos_df['Endereço Completo'] == endereco, 'Latitude'].values[0],
                  pedidos_df.loc[pedidos_df['Endereço Completo'] == endereco, 'Longitude'].values[0])
        G.add_node(endereco, pos=coords)
    for (endereco1, endereco2) in permutations([endereco_partida] + list(enderecos), 2):
        coords_1 = G.nodes[endereco1]['pos']
        coords_2 = G.nodes[endereco2]['pos']
        distancia = calcular_distancia(coords_1, coords_2)
        if distancia is not None:
            G.add_edge(endereco1, endereco2, weight=distancia)
    return G

def resolver_tsp_genetico(G):
    def fitness(route):
        return sum(G.edges[route[i], route[i+1]]['weight'] for i in range(len(route) - 1)) + G.edges[route[-1], route[0]]['weight']
    def mutate(route):
        i, j = random.sample(range(len(route)), 2)
        route[i], route[j] = route[j], route[i]
        return route
    def crossover(route1, route2):
        size = len(route1)
        start, end = sorted(random.sample(range(size), 2))
        child = [None] * size
        child[start:end] = route1[start:end]
        pointer = 0
        for i in range(size):
            if route2[i] not in child:
                while child[pointer] is not None:
                    pointer += 1
                child[pointer] = route2[i]
        return child
    def genetic_algorithm(population, generations=1000, mutation_rate=0.01):
        for _ in range(generations):
            population = sorted(population, key=lambda route: fitness(route))
            next_generation = population[:2]
            for _ in range(len(population) // 2 - 1):
                parents = random.sample(population[:10], 2)
                child = crossover(parents[0], parents[1])
                if random.random() < mutation_rate:
                    child = mutate(child)
                next_generation.append(child)
            population = next_generation
        return population[0], fitness(population[0])
    nodes = list(G.nodes)
    population = [random.sample(nodes, len(nodes)) for _ in range(100)]
    best_route, best_distance = genetic_algorithm(population)
    return best_route, best_distance

def agrupar_por_regiao(pedidos_df, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters)
    pedidos_df['Regiao'] = kmeans.fit_predict(pedidos_df[['Latitude', 'Longitude']])
    return pedidos_df

def criar_grafo_rotas(pedidos_df):
    """
    Cria um grafo de rotas a partir dos pedidos com coordenadas.
    """
    G = nx.Graph()
    for i, row1 in pedidos_df.iterrows():
        for j, row2 in pedidos_df.iterrows():
            if i != j:
                coord1 = (row1['Latitude'], row1['Longitude'])
                coord2 = (row2['Latitude'], row2['Longitude'])
                distancia = geodesic(coord1, coord2).kilometers
                G.add_edge(i, j, weight=distancia)
    return G

def calcular_rota_otima(grafo, ponto_partida):
    """
    Calcula a rota ótima usando o algoritmo de caminho mínimo.
    """
    try:
        caminho = nx.approximation.traveling_salesman_problem(
            grafo, cycle=True, weight='weight', method=nx.approximation.greedy_tsp
        )
        return caminho
    except Exception as e:
        raise ValueError(f"Erro ao calcular a rota ótima: {e}")

def obter_distancia_total(grafo, rota):
    """
    Calcula a distância total de uma rota.
    """
    distancia_total = 0
    for i in range(len(rota) - 1):
        distancia_total += grafo[rota[i]][rota[i + 1]]['weight']
    return distancia_total

def dummy_route_solver(pedidos, frota):
    # Esta função é um placeholder para o algoritmo de roteirização
    # Implemente aqui o VRP usando OR-Tools ou outro algoritmo
    return pedidos  # Retorna os pedidos sem alteração por enquanto

def resolver_vrp(pedidos_df, frota):
    """
    Resolve o problema de roteirização de veículos (VRP) usando OR-Tools.
    """
    # Preparar dados para o VRP
    coordenadas = pedidos_df[['Latitude', 'Longitude']].values
    num_pedidos = len(coordenadas)
    num_veiculos = len(frota)
    matriz_distancias = [
        [geodesic(coordenadas[i], coordenadas[j]).meters for j in range(num_pedidos)]
        for i in range(num_pedidos)
    ]

    # Criar o gerenciador de dados
    gerenciador = pywrapcp.RoutingIndexManager(len(matriz_distancias), num_veiculos, 0)

    # Criar o modelo de roteirização
    roteirizador = pywrapcp.RoutingModel(gerenciador)

    # Função de custo (distância)
    def distancia_callback(from_index, to_index):
        from_node = gerenciador.IndexToNode(from_index)
        to_node = gerenciador.IndexToNode(to_index)
        return matriz_distancias[from_node][to_node]

    transit_callback_index = roteirizador.RegisterTransitCallback(distancia_callback)
    roteirizador.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Restrições de capacidade (se aplicável)
    # Exemplo: roteirizador.AddDimension(...)

    # Configurar parâmetros de busca
    parametros_busca = pywrapcp.DefaultRoutingSearchParameters()
    parametros_busca.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )

    # Resolver o problema
    solucao = roteirizador.SolveWithParameters(parametros_busca)

    if solucao:
        rotas = []
        for veiculo_id in range(num_veiculos):
            indice = roteirizador.Start(veiculo_id)
            rota = []
            while not roteirizador.IsEnd(indice):
                node = gerenciador.IndexToNode(indice)
                rota.append(node)
                indice = solucao.Value(roteirizador.NextVar(indice))
            rotas.append(rota)
        return rotas
    else:
        raise ValueError("Não foi possível encontrar uma solução para o VRP.")

def tsp_nearest_neighbor(pedidos_df, partida_coords):
    """
    Algoritmo TSP simples (Nearest Neighbor) para um único veículo.
    """
    coords = [(row['Latitude'], row['Longitude']) for _, row in pedidos_df.iterrows()]
    n = len(coords)
    visitados = [False] * n
    rota = []
    atual = 0
    # Começa pelo ponto mais próximo do local de partida
    dists = [geodesic(partida_coords, c).meters for c in coords]
    atual = dists.index(min(dists))
    rota.append(atual)
    visitados[atual] = True
    for _ in range(n - 1):
        menor = float('inf')
        prox = None
        for i, c in enumerate(coords):
            if not visitados[i]:
                dist = geodesic(coords[atual], c).meters
                if dist < menor:
                    menor = dist
                    prox = i
        rota.append(prox)
        visitados[prox] = True
        atual = prox
    return rota

def tsp_genetico(pedidos_df, partida_coords, generations=500, pop_size=100, mutation_rate=0.02):
    """
    Algoritmo TSP usando Algoritmo Genético.
    """
    coords = [(row['Latitude'], row['Longitude']) for _, row in pedidos_df.iterrows()]
    n = len(coords)
    def fitness(route):
        dist = geodesic(partida_coords, coords[route[0]]).meters
        for i in range(n - 1):
            dist += geodesic(coords[route[i]], coords[route[i+1]]).meters
        dist += geodesic(coords[route[-1]], partida_coords).meters
        return dist
    def mutate(route):
        i, j = random.sample(range(n), 2)
        route[i], route[j] = route[j], route[i]
        return route
    def crossover(r1, r2):
        start, end = sorted(random.sample(range(n), 2))
        child = [None]*n
        child[start:end] = r1[start:end]
        pointer = 0
        for i in range(n):
            if r2[i] not in child:
                while child[pointer] is not None:
                    pointer += 1
                child[pointer] = r2[i]
        return child
    population = [random.sample(range(n), n) for _ in range(pop_size)]
    for _ in range(generations):
        population = sorted(population, key=fitness)
        next_gen = population[:2]
        for _ in range(pop_size//2 - 1):
            parents = random.sample(population[:10], 2)
            child = crossover(parents[0], parents[1])
            if random.random() < mutation_rate:
                child = mutate(child)
            next_gen.append(child)
        population = next_gen
    best = min(population, key=fitness)
    return best

def resolver_vrp(pedidos_df, frota_df):
    """
    VRP com OR-Tools para múltiplos veículos.
    """
    coords = [(row['Latitude'], row['Longitude']) for _, row in pedidos_df.iterrows()]
    n = len(coords)
    num_veiculos = len(frota_df)
    matriz = [[geodesic(coords[i], coords[j]).meters for j in range(n)] for i in range(n)]
    manager = pywrapcp.RoutingIndexManager(n, num_veiculos, 0)
    routing = pywrapcp.RoutingModel(manager)
    def dist_callback(from_idx, to_idx):
        return int(matriz[manager.IndexToNode(from_idx)][manager.IndexToNode(to_idx)])
    transit_callback_index = routing.RegisterTransitCallback(dist_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    solution = routing.SolveWithParameters(search_params)
    rotas = []
    if solution:
        for v in range(num_veiculos):
            idx = routing.Start(v)
            rota = []
            while not routing.IsEnd(idx):
                node = manager.IndexToNode(idx)
                rota.append(node)
                idx = solution.Value(routing.NextVar(idx))
            rotas.append(rota)
    return rotas

def resolver_vrp(pedidos_df, frota_df, capacidade_max=None):
    """
    VRP com OR-Tools para múltiplos veículos, com restrição de capacidade máxima individual por veículo.
    """
    coords = [(row['Latitude'], row['Longitude']) for _, row in pedidos_df.iterrows()]
    n = len(coords)
    num_veiculos = len(frota_df)
    matriz = [[geodesic(coords[i], coords[j]).meters for j in range(n)] for i in range(n)]
    manager = pywrapcp.RoutingIndexManager(n, num_veiculos, 0)
    routing = pywrapcp.RoutingModel(manager)
    def dist_callback(from_idx, to_idx):
        return int(matriz[manager.IndexToNode(from_idx)][manager.IndexToNode(to_idx)])
    transit_callback_index = routing.RegisterTransitCallback(dist_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Restrição de capacidade individual por veículo
    if 'Peso dos Itens' in pedidos_df.columns and 'Capac. Kg' in frota_df.columns:
        demands = [int(p) for p in pedidos_df['Peso dos Itens']]
        def demand_callback(from_index):
            node = manager.IndexToNode(from_index)
            return demands[node]
        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        capacidades = [int(c) for c in frota_df['Capac. Kg']]
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # sem capacidade extra
            capacidades,  # capacidade individual de cada veículo
            True,  # start cumul to zero
            'Capacity')

    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    solution = routing.SolveWithParameters(search_params)
    rotas = []
    if solution:
        for v in range(num_veiculos):
            idx = routing.Start(v)
            rota = []
            while not routing.IsEnd(idx):
                node = manager.IndexToNode(idx)
                rota.append(node)
                idx = solution.Value(routing.NextVar(idx))
            rotas.append(rota)
    return rotas

def resolver_vrp(pedidos_df, frota_df, percentual_utilizacao=100):
    """
    VRP com OR-Tools para múltiplos veículos, considerando a capacidade individual de cada veículo
    e um percentual de utilização definido pelo usuário.
    percentual_utilizacao: valor entre 0 e 100 (ex: 80 para usar até 80% da capacidade de cada veículo)
    """
    coords = [(row['Latitude'], row['Longitude']) for _, row in pedidos_df.iterrows()]
    n = len(coords)
    num_veiculos = len(frota_df)
    matriz = [[geodesic(coords[i], coords[j]).meters for j in range(n)] for i in range(n)]
    manager = pywrapcp.RoutingIndexManager(n, num_veiculos, 0)
    routing = pywrapcp.RoutingModel(manager)
    def dist_callback(from_idx, to_idx):
        return int(matriz[manager.IndexToNode(from_idx)][manager.IndexToNode(to_idx)])
    transit_callback_index = routing.RegisterTransitCallback(dist_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Restrição de capacidade individual por veículo com percentual
    if 'Peso dos Itens' in pedidos_df.columns and 'Capac. Kg' in frota_df.columns:
        demands = [int(p) for p in pedidos_df['Peso dos Itens']]
        def demand_callback(from_index):
            node = manager.IndexToNode(from_index)
            return demands[node]
        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        capacidades = [int(c * percentual_utilizacao / 100) for c in frota_df['Capac. Kg']]
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # sem capacidade extra
            capacidades,  # capacidade individual ajustada por %
            True,  # start cumul to zero
            'Capacity')

    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    solution = routing.SolveWithParameters(search_params)
    rotas = []
    if solution:
        for v in range(num_veiculos):
            idx = routing.Start(v)
            rota = []
            while not routing.IsEnd(idx):
                node = manager.IndexToNode(idx)
                rota.append(node)
                idx = solution.Value(routing.NextVar(idx))
            rotas.append(rota)
    return rotas

def agrupar_por_regiao(pedidos_df, n_clusters):
    """
    Agrupa pedidos por proximidade geográfica usando KMeans.
    """
    kmeans = KMeans(n_clusters=n_clusters)
    pedidos_df['Regiao'] = kmeans.fit_predict(pedidos_df[['Latitude', 'Longitude']])
    return pedidos_df

def exportar_rotas_para_planilhas(pedidos_df, rotas, pasta_saida='src/database/rotas_exportadas'):
    os.makedirs(pasta_saida, exist_ok=True)
    arquivos_gerados = []
    for veiculo_id, rota in enumerate(rotas):
        pedidos_rota = pedidos_df.iloc[rota]
        total_entregas = pedidos_rota['Qtde. dos Itens'].sum() if 'Qtde. dos Itens' in pedidos_rota else None
        total_peso = pedidos_rota['Peso dos Itens'].sum() if 'Peso dos Itens' in pedidos_rota else None
        pedidos_rota = pedidos_rota.copy()
        pedidos_rota['Total Entregas'] = total_entregas
        pedidos_rota['Total Peso'] = total_peso
        nome_arquivo = f'rota_veiculo_{veiculo_id+1}.xlsx'
        caminho_arquivo = os.path.join(pasta_saida, nome_arquivo)
        pedidos_rota.to_excel(caminho_arquivo, index=False)
        arquivos_gerados.append((nome_arquivo, caminho_arquivo))
    return arquivos_gerados

def criar_mapa_rotas(pedidos_df, rotas=None, partida_coords=(-23.0838, -47.1336)):
    """
    Visualização de rotas e pontos no mapa usando Folium.
    """
    mapa = folium.Map(location=partida_coords, zoom_start=11)
    folium.Marker(partida_coords, popup="Partida", icon=folium.Icon(color='red')).add_to(mapa)
    for _, row in pedidos_df.iterrows():
        folium.Marker([row['Latitude'], row['Longitude']], popup=row.get('Nome Cliente', ''), icon=folium.Icon(color='blue')).add_to(mapa)
    if rotas:
        cores = ['blue', 'green', 'purple', 'orange', 'darkred', 'cadetblue']
        for i, rota in enumerate(rotas):
            pontos = [partida_coords] + [(pedidos_df.iloc[idx]['Latitude'], pedidos_df.iloc[idx]['Longitude']) for idx in rota]
            folium.PolyLine(pontos, color=cores[i % len(cores)], weight=3, opacity=0.7).add_to(mapa)
    return mapa

def roteirizacao_manual(pedidos_df):
    """
    Permite ao usuário selecionar manualmente a ordem dos pedidos (exemplo para Streamlit).
    """
    st = None
    try:
        import streamlit as st
    except ImportError:
        return pedidos_df
    st.write("Arraste para ordenar os pedidos manualmente:")
    pedidos_df = st.data_editor(pedidos_df, use_container_width=True, num_rows="dynamic")
    return pedidos_df

def pre_processamento_frota_regiao(pedidos_df, frota_df, n_clusters=5):
    """
    Pré-processa os pedidos agrupando por região, calcula o peso total de cada região,
    seleciona veículos suficientes para cada região e distribui os pedidos respeitando capacidade e sem repetir clientes.
    Retorna um DataFrame com coluna 'Veiculo' e 'Placa' já preenchidas.
    """
    from sklearn.cluster import KMeans
    import numpy as np
    # Agrupar por região
    pedidos_df = pedidos_df.copy()
    pedidos_df = pedidos_df.dropna(subset=['Latitude', 'Longitude', 'Peso dos Itens'])
    pedidos_df = pedidos_df[pedidos_df['Peso dos Itens'] > 0]
    pedidos_df['Regiao'] = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(pedidos_df[['Latitude', 'Longitude']])
    pedidos_df['Veiculo'] = None
    pedidos_df['Placa'] = None
    frota_df = frota_df.sort_values('Capac. Kg', ascending=False).reset_index(drop=True)
    veiculo_global = 1
    for regiao in pedidos_df['Regiao'].unique():
        pedidos_regiao = pedidos_df[pedidos_df['Regiao'] == regiao].copy()
        peso_total = pedidos_regiao['Peso dos Itens'].sum()
        veiculos_necessarios = []
        peso_restante = peso_total
        frota_disp = frota_df.copy()
        while peso_restante > 0 or cx_restante > 0 and not frota_disp.empty:
            veiculo = frota_disp.iloc[0]
            veiculos_necessarios.append(veiculo)
            peso_restante -= veiculo['Capac. Kg']
            frota_disp = frota_disp.iloc[1:]
        if peso_restante > 0 or cx_restante > 0:
            st.warning(f"Região {regiao} excede a capacidade da frota disponível! Alguns pedidos podem ficar sem alocação.")
        # Distribuir pedidos entre veículos sem repetir clientes
        pedidos_regiao = pedidos_regiao.sort_values('Peso dos Itens', ascending=False)
        clientes_alocados = set()
        idx_pedido = 0
        for veiculo_idx, veiculo in enumerate(veiculos_necessarios):
            capacidade_kg = veiculo['Capac. Kg']
            capacidade_cx = veiculo['Capac. Cx'] if 'Capac. Cx' in veiculo else np.inf
            peso_usado = 0
            cx_usado = 0
            while idx_pedido < len(pedidos_regiao):
                pedido = pedidos_regiao.iloc[idx_pedido]
                cod_cliente = pedido['Cód. Cliente'] if 'Cód. Cliente' in pedido else pedido['Nome Cliente']
                if cod_cliente in clientes_alocados:
                    idx_pedido += 1
                    continue
                if (peso_usado + pedido['Peso dos Itens'] <= capacidade_kg) and (cx_usado + pedido.get('Qtde. dos Itens', 0) <= capacidade_cx):
                    pedidos_df_idx = pedidos_regiao.index[idx_pedido]
                    pedidos_df.at[pedidos_df_idx, 'Veiculo'] = veiculo_global
                    pedidos_df.at[pedidos_df_idx, 'Placa'] = veiculo['Placa']
                    peso_usado += pedido['Peso dos Itens']
                    cx_usado += pedido.get('Qtde. dos Itens', 0)
                    clientes_alocados.add(cod_cliente)
                    idx_pedido += 1
                else:
                    break
            veiculo_global += 1
    return pedidos_df

import numpy as np
from sklearn.cluster import KMeans

def pre_processamento_inteligente(pedidos_df, frota_df, n_clusters=5, prioridade_col=None, clientes_vip=None, entregas_urgentes=None):
    """
    Pré-processamento inteligente para roteirização:
    - Agrupa pedidos por região (KMeans)
    - Calcula peso total de cada região
    - Seleciona veículos suficientes para cada região, priorizando maior capacidade
    - Distribui pedidos entre veículos, sem repetir clientes
    - Permite priorizar clientes VIP ou entregas urgentes
    - Valida restrições e sugere ações ao usuário
    - Retorna DataFrame com colunas 'Veiculo', 'Placa', 'Regiao', 'Prioridade', 'Status Alocacao'
    - Exibe resumo visual do aproveitamento da frota
    """
    import streamlit as st
    pedidos_df = pedidos_df.copy()
    frota_df = frota_df.copy()
    # Checagem de dados
    if pedidos_df.empty or frota_df.empty:
        st.error("Pedidos ou frota vazios!")
        return pedidos_df
    if 'Peso dos Itens' not in pedidos_df.columns or 'Qtde. dos Itens' not in pedidos_df.columns:
        st.error("Pedidos precisam ter as colunas 'Peso dos Itens' e 'Qtde. dos Itens'.")
        return pedidos_df
    if 'Capac. Kg' not in frota_df.columns or 'Capac. Cx' not in frota_df.columns:
        st.error("Frota precisa ter as colunas 'Capac. Kg' e 'Capac. Cx'.")
        return pedidos_df
    pedidos_df = pedidos_df.dropna(subset=['Latitude', 'Longitude', 'Peso dos Itens', 'Qtde. dos Itens'])
    pedidos_df = pedidos_df[(pedidos_df['Peso dos Itens'] > 0) & (pedidos_df['Qtde. dos Itens'] > 0)]
    frota_df = frota_df[(frota_df['Capac. Kg'] > 0) & (frota_df['Capac. Cx'] > 0)]
    if pedidos_df.empty or frota_df.empty:
        st.error("Pedidos ou frota sem dados válidos!")
        return pedidos_df
    # Agrupamento por região
    pedidos_df['Regiao'] = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(pedidos_df[['Latitude', 'Longitude']])
    pedidos_df['Veiculo'] = None
    pedidos_df['Placa'] = None
    pedidos_df['Prioridade'] = 'Normal'
    pedidos_df['Status Alocacao'] = 'Não Alocado'
    if prioridade_col and prioridade_col in pedidos_df.columns:
        pedidos_df['Prioridade'] = pedidos_df[prioridade_col]
    if clientes_vip:
        pedidos_df.loc[pedidos_df['Cód. Cliente'].isin(clientes_vip), 'Prioridade'] = 'VIP'
    if entregas_urgentes:
        pedidos_df.loc[pedidos_df['Cód. Cliente'].isin(entregas_urgentes), 'Prioridade'] = 'Urgente'
    resumo_frota = []
    veiculo_global = 1
    pedidos_nao_alocados = []
    for regiao in pedidos_df['Regiao'].unique():
        pedidos_regiao = pedidos_df[pedidos_df['Regiao'] == regiao].copy()
        peso_total = pedidos_regiao['Peso dos Itens'].sum()
        cx_total = pedidos_regiao['Qtde. dos Itens'].sum()
        # Seleciona veículos suficientes para a região
        frota_disp = frota_df.copy().sort_values('Capac. Kg', ascending=False).reset_index(drop=True)
        veiculos_necessarios = []
        peso_restante = peso_total
        cx_restante = cx_total
        while (peso_restante > 0 or cx_restante > 0) and not frota_disp.empty:
            veiculo = frota_disp.iloc[0]
            veiculos_necessarios.append(veiculo)
            peso_restante -= veiculo['Capac. Kg']
            cx_restante -= veiculo['Capac. Cx']
            frota_disp = frota_disp.iloc[1:]
        if peso_restante > 0 or cx_restante > 0:
            st.warning(f"Região {regiao} excede a capacidade da frota disponível! Alguns pedidos podem ficar sem alocação.")
        # Distribuição de pedidos entre veículos
        pedidos_regiao = pedidos_regiao.sort_values(['Prioridade', 'Peso dos Itens'], ascending=[True, False])
        clientes_alocados = set()
        idx_pedido = 0
        for veiculo_idx, veiculo in enumerate(veiculos_necessarios):
            capacidade_kg = veiculo['Capac. Kg']
            capacidade_cx = veiculo['Capac. Cx']
            peso_usado = 0
            cx_usado = 0
            while idx_pedido < len(pedidos_regiao):
                pedido = pedidos_regiao.iloc[idx_pedido]
                cod_cliente = pedido['Cód. Cliente'] if 'Cód. Cliente' in pedido else pedido['Nome Cliente']
                if cod_cliente in clientes_alocados:
                    idx_pedido += 1
                    continue
                if (peso_usado + pedido['Peso dos Itens'] <= capacidade_kg) and (cx_usado + pedido['Qtde. dos Itens'] <= capacidade_cx):
                    pedidos_df_idx = pedidos_regiao.index[idx_pedido]
                    pedidos_df.at[pedidos_df_idx, 'Veiculo'] = veiculo_global
                    pedidos_df.at[pedidos_df_idx, 'Placa'] = veiculo['Placa']
                    pedidos_df.at[pedidos_df_idx, 'Status Alocacao'] = 'Alocado'
                    peso_usado += pedido['Peso dos Itens']
                    cx_usado += pedido['Qtde. dos Itens']
                    clientes_alocados.add(cod_cliente)
                    idx_pedido += 1
                else:
                    break
            resumo_frota.append({
                'Regiao': regiao,
                'Veiculo': veiculo_global,
                'Placa': veiculo['Placa'],
                'Peso Alocado': peso_usado,
                'Caixas Alocadas': cx_usado,
                'Capacidade Peso': capacidade_kg,
                'Capacidade Caixas': capacidade_cx,
                'Aproveitamento Peso (%)': round(100*peso_usado/capacidade_kg,1) if capacidade_kg else 0,
                'Aproveitamento Caixas (%)': round(100*cx_usado/capacidade_cx,1) if capacidade_cx else 0
            })
            veiculo_global += 1
        # Marcar pedidos não alocados
        for idx in pedidos_regiao.index:
            if pedidos_df.at[idx, 'Status Alocacao'] != 'Alocado':
                pedidos_nao_alocados.append(idx)
    # Resumo visual do aproveitamento da frota
    resumo_frota_df = pd.DataFrame(resumo_frota)
    st.subheader("Resumo do Aproveitamento da Frota")
    st.dataframe(resumo_frota_df, use_container_width=True)
    if pedidos_nao_alocados:
        st.warning(f"{len(pedidos_nao_alocados)} pedidos não foram alocados. Considere aumentar a frota ou dividir a entrega em mais dias.")
    return pedidos_df

def alocacao_prioridade_capacidade_regiao(pedidos_df, frota_df, n_clusters=5):
    """
    Aloca pedidos considerando simultaneamente a capacidade dos veículos e a região (cluster).
    1. Agrupa pedidos por região (KMeans).
    2. Para cada região, aloca pedidos nos veículos disponíveis, respeitando a capacidade individual de cada veículo.
    3. Tenta preencher cada veículo ao máximo dentro da sua região antes de passar para outra região.
    Retorna o DataFrame de pedidos com colunas 'Veiculo', 'Placa', 'Regiao', 'Status Alocacao'.
    """
    import numpy as np
    from sklearn.cluster import KMeans
    pedidos_df = pedidos_df.copy()
    frota_df = frota_df.copy()
    pedidos_df = pedidos_df.dropna(subset=['Latitude', 'Longitude', 'Peso dos Itens'])
    pedidos_df = pedidos_df[pedidos_df['Peso dos Itens'] > 0]
    frota_df = frota_df[frota_df['Capac. Kg'] > 0]
    pedidos_df['Regiao'] = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(pedidos_df[['Latitude', 'Longitude']])
    pedidos_df['Veiculo'] = None
    pedidos_df['Placa'] = None
    pedidos_df['Status Alocacao'] = 'Não Alocado'
    frota_df = frota_df.sort_values('Capac. Kg', ascending=False).reset_index(drop=True)
    veiculo_global = 1
    for regiao in pedidos_df['Regiao'].unique():
        pedidos_regiao = pedidos_df[pedidos_df['Regiao'] == regiao].copy()
        pedidos_regiao = pedidos_regiao.sort_values('Peso dos Itens', ascending=False)
        idx_pedido = 0
        frota_disp = frota_df.copy()
        for veiculo_idx, veiculo in frota_disp.iterrows():
            capacidade_kg = veiculo['Capac. Kg']
            capacidade_cx = veiculo['Capac. Cx'] if 'Capac. Cx' in veiculo else np.inf
            peso_usado = 0
            cx_usado = 0
            while idx_pedido < len(pedidos_regiao):
                pedido = pedidos_regiao.iloc[idx_pedido]
                if (peso_usado + pedido['Peso dos Itens'] <= capacidade_kg) and (cx_usado + pedido.get('Qtde. dos Itens', 0) <= capacidade_cx):
                    pedidos_df_idx = pedidos_regiao.index[idx_pedido]
                    pedidos_df.at[pedidos_df_idx, 'Veiculo'] = veiculo_global
                    pedidos_df.at[pedidos_df_idx, 'Placa'] = veiculo['Placa']
                    pedidos_df.at[pedidos_df_idx, 'Status Alocacao'] = 'Alocado'
                    peso_usado += pedido['Peso dos Itens']
                    cx_usado += pedido.get('Qtde. dos Itens', 0)
                    idx_pedido += 1
                else:
                    break
            veiculo_global += 1
    return pedidos_df

def alocacao_regioes_por_veiculo(pedidos_df, frota_df, n_clusters=3, regioes_por_veiculo=1):
    """
    Aloca pedidos para veículos considerando:
    - Agrupamento dos pedidos em regiões (clusters)
    - Cada veículo pode atender até 'regioes_por_veiculo' diferentes
    - Todos os pedidos devem ser roteirizados
    - Respeita a capacidade individual de cada veículo (Capac. Kg)
    Retorna DataFrame de pedidos com colunas 'Veiculo', 'Placa', 'Regiao', 'Status Alocacao'.
    """
    import numpy as np
    from sklearn.cluster import KMeans
    from itertools import combinations
    pedidos_df = pedidos_df.copy()
    frota_df = frota_df.copy()
    pedidos_df = pedidos_df.dropna(subset=['Latitude', 'Longitude', 'Peso dos Itens'])
    pedidos_df = pedidos_df[pedidos_df['Peso dos Itens'] > 0]
    frota_df = frota_df[frota_df['Capac. Kg'] > 0]
    pedidos_df['Regiao'] = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(pedidos_df[['Latitude', 'Longitude']])
    pedidos_df['Veiculo'] = None
    pedidos_df['Placa'] = None
    pedidos_df['Status Alocacao'] = 'Não Alocado'
    frota_df = frota_df.sort_values('Capac. Kg', ascending=False).reset_index(drop=True)
    veiculo_global = 1
    regioes = pedidos_df['Regiao'].unique().tolist()
    regioes_combinacoes = list(combinations(regioes, min(regioes_por_veiculo, len(regioes))))
    veiculo_idx = 0
    pedidos_restantes = pedidos_df[pedidos_df['Status Alocacao'] == 'Não Alocado']
    while not pedidos_restantes.empty and veiculo_idx < len(frota_df):
        veiculo = frota_df.iloc[veiculo_idx]
        capacidade_kg = veiculo['Capac. Kg']
        capacidade_cx = veiculo['Capac. Cx'] if 'Capac. Cx' in veiculo else np.inf
        peso_usado = 0
        cx_usado = 0
        regioes_veiculo = regioes_combinacoes[veiculo_idx % len(regioes_combinacoes)]
        pedidos_candidato = pedidos_restantes[pedidos_restantes['Regiao'].isin(regioes_veiculo)]
        for idx, pedido in pedidos_candidato.iterrows():
            if (peso_usado + pedido['Peso dos Itens'] <= capacidade_kg) and (cx_usado + pedido.get('Qtde. dos Itens', 0) <= capacidade_cx):
                pedidos_df.at[idx, 'Veiculo'] = veiculo_global
                pedidos_df.at[idx, 'Placa'] = veiculo['Placa']
                pedidos_df.at[idx, 'Status Alocacao'] = 'Alocado'
                peso_usado += pedido['Peso dos Itens']
                cx_usado += pedido.get('Qtde. dos Itens', 0)
        veiculo_global += 1
        veiculo_idx += 1
        pedidos_restantes = pedidos_df[pedidos_df['Status Alocacao'] == 'Não Alocado']
    # Se ainda restarem pedidos, tenta alocar respeitando capacidade
    if not pedidos_restantes.empty:
        for idx, pedido in pedidos_restantes.iterrows():
            for veiculo_idx, veiculo in frota_df.iterrows():
                capacidade_kg = veiculo['Capac. Kg']
                capacidade_cx = veiculo['Capac. Cx'] if 'Capac. Cx' in veiculo else np.inf
                peso_alocado = pedidos_df[(pedidos_df['Placa'] == veiculo['Placa']) & (pedidos_df['Status Alocacao'] == 'Alocado')]['Peso dos Itens'].sum()
                cx_alocado = pedidos_df[(pedidos_df['Placa'] == veiculo['Placa']) & (pedidos_df['Status Alocacao'] == 'Alocado')]['Qtde. dos Itens'].sum() if 'Qtde. dos Itens' in pedidos_df.columns else 0
                if (peso_alocado + pedido['Peso dos Itens'] <= capacidade_kg) and (cx_alocado + pedido.get('Qtde. dos Itens', 0) <= capacidade_cx):
                    pedidos_df.at[idx, 'Veiculo'] = veiculo_idx + 1
                    pedidos_df.at[idx, 'Placa'] = veiculo['Placa']
                    pedidos_df.at[idx, 'Status Alocacao'] = 'Alocado'
                    break
    return pedidos_df
