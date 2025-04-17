import streamlit as st
st.set_page_config(page_title="Roteirizador de Entregas", layout="wide")
from dashboard_frota import dashboard_frota
from dashboard_pedidos import dashboard_pedidos
from dashboard_ia import dashboard_ia
from dashboard_routing import dashboard_routing

def main():
    st.markdown("""
    <style>
    .sidebar-menu-custom {
        background: linear-gradient(180deg, #e3f2fd 0%, #bbdefb 100%);
        border-radius: 16px;
        padding: 1.5em 1em 1em 1em;
        margin-bottom: 2em;
        box-shadow: 0 2px 8px rgba(26,115,232,0.08);
    }
    .sidebar-menu-custom label, .sidebar-menu-custom span, .sidebar-menu-custom div[data-testid="stRadio"] label {
        font-size: 1.15em !important;
        font-weight: 600;
        color: #1a237e !important;
    }
    .sidebar-menu-custom .stRadio > div {
        gap: 0.5em;
    }
    .sidebar-menu-custom .stRadio label {
        padding: 0.3em 0.7em;
        border-radius: 8px;
        transition: background 0.2s;
    }
    .sidebar-menu-custom .stRadio label[data-selected="true"] {
        background: #1976d2;
        color: #fff !important;
    }
    </style>
    <div class='sidebar-menu-custom'>
        <span style='font-size:1.3em;font-weight:bold;color:#1976d2;'>Menu</span>
    </div>
    """, unsafe_allow_html=True)
    with st.sidebar:
        menu = st.radio(
            "",
            ("Dashboard Frota", "Dashboard Pedidos", "Dashboard Routing", "Dashboard IA"),
            key="main_menu_radio"
        )
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