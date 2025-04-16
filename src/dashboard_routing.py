import streamlit as st
import pandas as pd
import os
from routing import resolver_vrp

def dashboard_routing():
    st.header("Dashboard de Roteirização de Pedidos")

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

    # Resolver o problema de roteirização
    try:
        rotas = resolver_vrp(pedidos_df, frota_df)
        st.success("Roteirização concluída com sucesso!")

        # Exibir rotas
        for veiculo_id, rota in enumerate(rotas):
            st.subheader(f"Rota para Veículo {veiculo_id + 1}")
            rota_enderecos = [pedidos_df.iloc[node]['Endereço de Entrega'] for node in rota]
            st.write(" -> ".join(rota_enderecos))

    except Exception as e:
        st.error(f"Erro ao realizar a roteirização: {e}")
