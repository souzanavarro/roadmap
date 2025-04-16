import networkx as nx
from itertools import permutations
import random
from geopy.distance import geodesic
from sklearn.cluster import KMeans
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

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
