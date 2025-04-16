import folium

def plot_points_on_map(df, lat_col='Latitude', lon_col='Longitude'):
    if df.empty:
        return None
    m = folium.Map(location=[df[lat_col].mean(), df[lon_col].mean()], zoom_start=12)
    for _, row in df.iterrows():
        folium.Marker([row[lat_col], row[lon_col]], popup=row.get('Nome Cliente', '')).add_to(m)
    return m

def criar_mapa(pedidos_df, endereco_partida_coords):
    import folium
    mapa = folium.Map(location=endereco_partida_coords, zoom_start=12)
    for _, row in pedidos_df.iterrows():
        popup_text = f"<b>Placa: {row['Placa']}</b><br>Endereço: {row['Endereço Completo']}"
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=popup_text,
            icon=folium.Icon(color='blue')
        ).add_to(mapa)
    folium.Marker(
        location=endereco_partida_coords,
        popup="Endereço de Partida",
        icon=folium.Icon(color='red')
    ).add_to(mapa)
    return mapa
