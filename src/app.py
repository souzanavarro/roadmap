import streamlit as st
st.set_page_config(page_title="Roteirizador de Entregas", layout="wide")
from dashboard_frota import dashboard_frota
from dashboard_pedidos import dashboard_pedidos
from dashboard_ia import dashboard_ia
from dashboard_routing import dashboard_routing

def main():
    st.sidebar.title("Menu")
    menu = st.sidebar.radio("Escolha uma opção:", ("Dashboard Frota", "Dashboard Pedidos", "Dashboard Routing", "Dashboard IA"))

    if menu == "Dashboard Frota":
        dashboard_frota()
    elif menu == "Dashboard Pedidos":
        dashboard_pedidos()
    elif menu == "Dashboard Routing":
        dashboard_routing()
    elif menu == "Dashboard IA":
        dashboard_ia()

if __name__ == "__main__":
    main()