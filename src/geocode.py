from geopy.geocoders import Nominatim
import requests
import streamlit as st

def geocode_address(address):
    geolocator = Nominatim(user_agent="delivery_router")
    location = geolocator.geocode(address)
    if location:
        return location.latitude, location.longitude
    return None, None

def obter_coordenadas_opencage(endereco, api_keys):
    for api_key in api_keys:
        try:
            url = f"https://api.opencagedata.com/geocode/v1/json?q={endereco}&key={api_key}"
            response = requests.get(url)
            data = response.json()
            if 'status' in data and data['status']['code'] == 200 and 'results' in data:
                location = data['results'][0]['geometry']
                return (location['lat'], location['lng'])
            else:
                st.warning(f"API Key {api_key} falhou para o endereço: {endereco}. Tentando próxima chave...")
        except Exception as e:
            st.error(f"Erro ao tentar obter as coordenadas com a API Key {api_key}: {e}")
    st.error(f"Todas as APIs do OpenCage falharam para o endereço: {endereco}.")
    return None, None

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

def obter_coordenadas_com_fallback(endereco, coordenadas_salvas, api_keys):
    if endereco in coordenadas_salvas:
        return coordenadas_salvas[endereco]

    # Tentar primeiro com OpenCage
    coords = obter_coordenadas_opencage(endereco, api_keys)

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
