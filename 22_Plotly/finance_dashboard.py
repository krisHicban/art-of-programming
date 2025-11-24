# 💰 Dashboard Finanțe Complet cu Plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# Data reală de tracking financiar
expense_data = {
    'Locuinta': [1200, 1200, 1250, 1250, 1250, 1300],
    'Mancare': [680, 750, 820, 650, 900, 780],
    'Transport': [200, 150, 300, 180, 250, 220],
    'Entertainment': [300, 450, 500, 320, 600, 420],
    'Sanatate': [120, 80, 150, 200, 100, 180]
}

months = ['Ian', 'Feb', 'Mar', 'Apr', 'Mai', 'Iun']

# Subplot-uri pentru dashboard complet
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('📊 Trend Cheltuieli', '🥧 Breakdown Categorii', 
                   '📈 Comparație cu Buget', '⚠️ Alertă Categorii'),
    specs=[[{"type": "scatter"}, {"type": "pie"}],
           [{"type": "bar"}, {"type": "scatter"}]]
)

# 1. Trend line interactiv pentru fiecare categorie
for category, values in expense_data.items():
    fig.add_trace(
        go.Scatter(x=months, y=values, name=category,
                  mode='lines+markers',
                  line=dict(width=3),
                  hovertemplate='%{fullData.name}<br>%{y}€<br><extra></extra>'),
        row=1, col=1
    )

# 2. Pie chart interactiv pentru ultima lună
last_month_total = {cat: vals[-1] for cat, vals in expense_data.items()}
fig.add_trace(
    go.Pie(values=list(last_month_total.values()),
           labels=list(last_month_total.keys()),
           name="Breakdown Iun",
           hovertemplate='%{label}<br>€%{value}<br>%{percent}<extra></extra>'),
    row=1, col=2
)

# 3. Comparație cu bugetul planificat
budget = {'Locuinta': 1300, 'Mancare': 700, 'Transport': 200, 
          'Entertainment': 400, 'Sanatate': 150}

categories = list(budget.keys())
actual = [expense_data[cat][-1] for cat in categories]
planned = list(budget.values())

fig.add_trace(go.Bar(x=categories, y=planned, name='Buget Planificat',
                    marker_color='lightblue'), row=2, col=1)
fig.add_trace(go.Bar(x=categories, y=actual, name='Cheltuit Real',
                    marker_color='darkred'), row=2, col=1)

# 4. Alertă pentru categoriile cu creștere > 20%
growth_rates = []
alert_categories = []
for cat in categories:
    if len(expense_data[cat]) >= 2:
        growth = ((expense_data[cat][-1] - expense_data[cat][-2]) / expense_data[cat][-2]) * 100
        growth_rates.append(growth)
        if growth > 20:
            alert_categories.append(cat)

fig.add_trace(
    go.Scatter(x=categories, y=growth_rates, 
              mode='markers+text',
              marker=dict(size=15, color=['red' if cat in alert_categories else 'green' 
                                        for cat in categories]),
              text=[f'{rate:.1f}%' for rate in growth_rates],
              textposition="top center",
              name='Creștere %'),
    row=2, col=2
)

# Layout pentru dashboard profesional
fig.update_layout(
    title_text="💰 Personal Finance Intelligence Dashboard",
    title_x=0.5,
    height=800,
    showlegend=True,
    hovermode='closest'
)

# Configurări specifice pentru fiecare subplot
fig.update_xaxes(title_text="Luna", row=2, col=1)
fig.update_yaxes(title_text="Suma (€)", row=2, col=1)
fig.update_xaxes(title_text="Categorie", row=2, col=2)
fig.update_yaxes(title_text="Creștere (%)", row=2, col=2)

fig.show()

# 🎯 REZULTAT: Dashboard complet interactiv care revelează:
# - Trend-uri pentru fiecare categorie de cheltuieli
# - Breakdown vizual pentru luna curentă
# - Comparație cu bugetul planificat
# - Alertă automată pentru categoriile problematice
# 
# 💡 BONUS: Salvează ca HTML pentru acces permanent
fig.write_html("finance_dashboard.html")