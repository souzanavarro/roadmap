import pandas as pd
from geopy.geocoders import Nominatim
import requests
import streamlit as st

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

def preencher_regioes_pedidos(pedidos_df):
    """
    Preenche a coluna 'Região' do DataFrame de pedidos usando as coordenadas (latitude, longitude)
    e a função obter_regiao_por_coordenada do geocode.py.
    """
    from geocode import obter_regiao_por_coordenada
    import time
    regioes = []
    for idx, row in pedidos_df.iterrows():
        lat, lon = row['Latitude'], row['Longitude']
        if pd.notnull(lat) and pd.notnull(lon):
            regiao = obter_regiao_por_coordenada(lat, lon)
            regioes.append(regiao)
            print(f"[LOG] Pedido {idx}: Região encontrada: {regiao}")
            time.sleep(1)  # Respeita o limite da API gratuita Nominatim
        else:
            regioes.append(None)
    pedidos_df['Região'] = regioes
    return pedidos_df
