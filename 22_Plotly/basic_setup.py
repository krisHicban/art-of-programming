# 🚀 Primul tău grafic Plotly interactiv
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Data realistă pentru finante personale
df = pd.DataFrame({
    'Luna': ['Ian', 'Feb', 'Mar', 'Apr', 'Mai', 'Iun'],
    'Venituri': [850, 650, 900, 1200, 750, 950],  # Freelance + venit variabil
    'Cheltuieli': [820, 590, 875, 1030, 720, 870]  # Spending urmat de income patterns
})

# Grafic interactiv cu hover tooltips
fig = px.bar(df, x='Luna', y=['Venituri', 'Cheltuieli'],
             title='💰 Finanțele Tale Interactive',
             hover_data={'value': ':,.0f€'},  # Format custom pentru hover
             color_discrete_map={'Venituri': '#10B981', 'Cheltuieli': '#EF4444'})

# Configurări pentru interactivitate maximă
fig.update_layout(
    hovermode='x unified',  # Hover pe întreaga coloană
    xaxis_title="Luna",
    yaxis_title="Suma (€)",
    legend_title="Categorie"
)

# Adaugă annotations pentru insight-uri
fig.add_annotation(
    x="Mar", y=3100,
    text="📈 Luna cu cele mai mari cheltuieli!",
    showarrow=True,
    arrowhead=2,
    arrowcolor="red"
)

# Show cu toate feature-urile interactive
fig.show()

# 💡 MAGIA: Fiecare element devine interactiv automat!
# - Hover pentru detalii
# - Click pe legend pentru hide/show
# - Zoom cu scroll wheel
# - Pan cu drag
# - Download ca PNG/HTML