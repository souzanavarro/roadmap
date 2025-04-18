import pandas as pd
from geopy.geocoders import Nominatim
import requests
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import pickle
import os

def geocode_address(address):
    geolocator = Nominatim(user_agent="delivery_router")
    location = geolocator.geocode(address)
    if location:
        return location.latitude, location.longitude
    return None, None

def obter_coordenadas_opencage(endereco):
    try:
        api_key = "6f522c67add14152926990afbe127384"  # Sua chave de API do OpenCage
        url = f"https://api.opencagedata.com/geocode/v1/json?q={endereco}&key={api_key}&language=pt-BR&countrycode=br&limit=1"
        response = requests.get(url)
        data = response.json()
        if (
            'status' in data and data['status']['code'] == 200 and
            'results' in data and len(data['results']) > 0 and
            'geometry' in data['results'][0]
        ):
            location = data['results'][0]['geometry']
            return (location['lat'], location['lng'])
        else:
            st.error(f"Não foi possível obter as coordenadas para o endereço: {endereco}. Status: {data.get('status', {}).get('message', 'Desconhecido')}")
            return None
    except Exception as e:
        st.error(f"Erro ao tentar obter as coordenadas: {e}")
        return None

def obter_coordenadas_nominatim(endereco):
    try:
        geolocator = Nominatim(user_agent="delivery_router")
        location = geolocator.geocode(endereco)
        if location:
            return location.latitude, location.longitude
        else:
            st.error(f"Não foi possível obter as coordenadas para o endereço: {endereco} usando Nominatim.")
            return None, None
    except Exception as e:
        st.error(f"Erro ao tentar obter as coordenadas com Nominatim: {e}")
        return None, None

def obter_coordenadas_com_fallback(endereco, coordenadas_salvas):
    if endereco in coordenadas_salvas:
        return coordenadas_salvas[endereco]

    # Tentar primeiro com OpenCage
    coords = obter_coordenadas_opencage(endereco)

    # Se OpenCage falhar, tentar com Nominatim
    if coords is None or coords == (None, None):
        coords = obter_coordenadas_nominatim(endereco)

    # Se ambos falharem, usar coordenadas manuais
    if coords is None or coords == (None, None):
        coordenadas_manuais = {
            "Rua Araújo Leite, 146, Centro, Piedade, São Paulo, Brasil": (-23.71241093449893, -47.41796911054548)
        }
        coords = coordenadas_manuais.get(endereco, (None, None))

    if coords and coords != (None, None):
        coordenadas_salvas[endereco] = coords
    return coords

def obter_regiao_por_coordenada(latitude, longitude):
    """
    Usa Nominatim (OpenStreetMap) para obter a região (bairro, cidade, estado) a partir de coordenadas.
    Retorna uma string com a melhor informação disponível.
    """
    try:
        from geopy.geocoders import Nominatim
        geolocator = Nominatim(user_agent="delivery_router_reverse")
        location = geolocator.reverse((latitude, longitude), language='pt-BR', addressdetails=True, timeout=10)
        if location and location.raw and 'address' in location.raw:
            address = location.raw['address']
            # Tenta pegar bairro, cidade e estado
            bairro = address.get('suburb') or address.get('neighbourhood') or address.get('quarter')
            cidade = address.get('city') or address.get('town') or address.get('village') or address.get('municipality')
            estado = address.get('state')
            pais = address.get('country')
            # Monta a string da região
            regiao = ', '.join([v for v in [bairro, cidade, estado, pais] if v])
            return regiao if regiao else None
        return None
    except Exception as e:
        print(f"[LOG] Erro ao buscar região por coordenada: {e}")
        return None

def _coord_hash(lat, lon, precision=5):
    """Gera um hash para coordenadas arredondadas para evitar duplicidade."""
    return hashlib.md5(f"{round(float(lat), precision)},{round(float(lon), precision)}".encode()).hexdigest()

def preencher_regioes_pedidos(pedidos_df, sleep_time=0.1, max_workers=8, cache_path="src/database/regioes_cache.pkl"):
    """
    Preenche a coluna 'Região' do DataFrame de pedidos usando as coordenadas (latitude, longitude)
    Utiliza cache local e paralelização para acelerar o processo.
    """
    from geocode import obter_regiao_por_coordenada
    import time
    regioes = [None] * len(pedidos_df)
    # Carregar cache local
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            regioes_cache = pickle.load(f)
    else:
        regioes_cache = {}
    # Função para processar uma linha
    def processa_linha(idx, lat, lon):
        if pd.isnull(lat) or pd.isnull(lon):
            return idx, None
        h = _coord_hash(lat, lon)
        if h in regioes_cache:
            return idx, regioes_cache[h]
        regiao = obter_regiao_por_coordenada(lat, lon)
        regioes_cache[h] = regiao
        return idx, regiao
    # Montar lista de tarefas
    tarefas = [(idx, row['Latitude'], row['Longitude']) for idx, row in pedidos_df.iterrows()]
    # Paralelizar
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futuros = {executor.submit(processa_linha, idx, lat, lon): idx for idx, lat, lon in tarefas}
        for fut in as_completed(futuros):
            idx, regiao = fut.result()
            regioes[idx] = regiao
    # Salvar cache atualizado
    with open(cache_path, "wb") as f:
        pickle.dump(regioes_cache, f)
    pedidos_df['Região'] = regioes
    return pedidos_df
