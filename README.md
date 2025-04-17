# Roadmap Roteirização - Streamlit App

Este projeto é uma aplicação de roteirização de entregas desenvolvida em Python com Streamlit, utilizando algoritmos de VRP (Vehicle Routing Problem), TSP, clustering e integração com mapas.

## Estrutura do Projeto

```
roadmap/
├── src/
│   ├── app.py                  # App principal Streamlit
│   ├── dashboard_routing.py    # Dashboard de roteirização
│   ├── dashboard_frota.py      # Dashboard de frota
│   ├── dashboard_pedidos.py    # Dashboard de pedidos
│   ├── dashboard_ia.py         # Dashboard IA
│   ├── routing.py              # Algoritmos de roteirização (VRP, TSP, clusters)
│   ├── geocode.py, map_utils.py
│   └── database/               # Dados e históricos
│       ├── database_pedidos.csv
│       ├── database_frota.csv
│       ├── historico_roteirizacoes.csv
│       └── rotas_exportadas/
├── requirements.txt            # Dependências Python
├── .devcontainer/devcontainer.json # Inicialização automática do Streamlit no Codespaces
└── README.md                   # Documentação
```

## Principais Funcionalidades
- Roteirização VRP com restrição de capacidade individual e percentual
- Alocação inteligente por região, capacidade, ou ambas
- Definição de clusters (regiões) por percentual
- Exportação de rotas para Excel
- Visualização de rotas em mapa (Folium)
- Dashboard interativo (Streamlit)
- Inicialização automática do Streamlit no Codespaces

## Instalação

1. Clone o repositório:
   ```bash
   git clone <repository-url>
   cd roadmap
   ```
2. (Opcional) Crie um ambiente virtual:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```
3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

## Executando o App

```bash
streamlit run src/app.py
```

No Codespaces, o Streamlit inicia automaticamente.

## Contribuição
Pull requests e sugestões são bem-vindos!

## Licença
MIT License