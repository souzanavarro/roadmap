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
