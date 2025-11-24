# 🏥 Health & Wellness Interactive Tracker
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generarea de date realiste pentru 2 săptămâni
dates = [datetime.now() - timedelta(days=13-i) for i in range(14)]
date_strings = [d.strftime('%d/%m') for d in dates]

# Data de health tracking
health_metrics = {
    'sleep_hours': [7.5, 6.8, 7.2, 8.0, 6.5, 7.8, 8.2, 7.1, 6.9, 7.6, 8.1, 7.4, 6.7, 7.9],
    'steps': [8500, 7200, 9100, 10200, 6800, 9500, 11200, 8800, 7600, 9800, 10500, 8900, 7400, 9200],
    'water_liters': [2.1, 1.8, 2.5, 2.8, 1.5, 2.3, 3.0, 2.2, 1.9, 2.6, 2.9, 2.4, 1.7, 2.5],
    'mood_score': [8, 6, 7, 9, 5, 8, 9, 7, 6, 8, 9, 8, 6, 8],
    'energy_level': [7, 5, 6, 9, 4, 8, 9, 6, 5, 8, 9, 7, 5, 7]
}

# Creare dashboard cu 3 secțiuni
fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=('🌙 Sleep Quality Heatmap', '👥 Correlations Matrix',
                   '🚶 Daily Activity Timeline', '💧 Hydration vs Energy',
                   '😊 Mood & Sleep Connection', '📊 Weekly Summary'),
    specs=[[{"type": "scatter"}, {"type": "scatter"}],
           [{"colspan": 2}, None],
           [{"type": "scatter"}, {"type": "bar"}]],
    vertical_spacing=0.08,
    horizontal_spacing=0.1
)

# 1. Sleep Quality Heatmap (simulat ca scatter cu culori)
sleep_colors = ['red' if h < 7 else 'yellow' if h < 8 else 'green' 
               for h in health_metrics['sleep_hours']]

fig.add_trace(
    go.Scatter(x=date_strings, y=['Sleep Quality']*14,
              mode='markers',
              marker=dict(size=25, color=health_metrics['sleep_hours'],
                         colorscale='RdYlGn', cmin=6, cmax=9,
                         colorbar=dict(title="Ore Somn")),
              text=[f'{h}h' for h in health_metrics['sleep_hours']],
              textposition="middle center",
              hovertemplate='Data: %{x}<br>Somn: %{text}<br><extra></extra>',
              name='Sleep Quality'),
    row=1, col=1
)

# 2. Correlation Matrix (sleep vs energy)
fig.add_trace(
    go.Scatter(x=health_metrics['sleep_hours'], y=health_metrics['energy_level'],
              mode='markers+text',
              marker=dict(size=12, color='blue', opacity=0.6),
              text=date_strings,
              textposition="top center",
              hovertemplate='Somn: %{x}h<br>Energie: %{y}/10<br>Data: %{text}<extra></extra>',
              name='Sleep vs Energy'),
    row=1, col=2
)

# Linie de trend pentru corelație
z = np.polyfit(health_metrics['sleep_hours'], health_metrics['energy_level'], 1)
p = np.poly1d(z)
fig.add_trace(
    go.Scatter(x=health_metrics['sleep_hours'], 
              y=p(health_metrics['sleep_hours']),
              mode='lines',
              line=dict(color='red', dash='dash'),
              name='Trend Line',
              hoverinfo='skip'),
    row=1, col=2
)

# 3. Daily Activity Timeline (full width)
fig.add_trace(
    go.Scatter(x=date_strings, y=health_metrics['steps'],
              mode='lines+markers',
              line=dict(color='purple', width=3),
              marker=dict(size=8),
              name='Steps Daily',
              hovertemplate='Data: %{x}<br>Pași: %{y:,}<br><extra></extra>'),
    row=2, col=1
)

# Target line pentru pași
fig.add_trace(
    go.Scatter(x=date_strings, y=[10000]*14,
              mode='lines',
              line=dict(color='green', dash='dash'),
              name='Target: 10k steps',
              hoverinfo='skip'),
    row=2, col=1
)

# 4. Hydration vs Energy scatter
fig.add_trace(
    go.Scatter(x=health_metrics['water_liters'], y=health_metrics['energy_level'],
              mode='markers',
              marker=dict(size=15, color=health_metrics['mood_score'],
                         colorscale='Blues', cmin=5, cmax=9),
              text=date_strings,
              hovertemplate='Apă: %{x}L<br>Energie: %{y}/10<br>Mood: %{marker.color}/10<br>Data: %{text}<extra></extra>',
              name='Water vs Energy'),
    row=3, col=1
)

# 5. Weekly Summary Bar Chart
weeks = ['Săpt 1', 'Săpt 2']
week1_avg = {
    'Sleep': np.mean(health_metrics['sleep_hours'][:7]),
    'Steps': np.mean(health_metrics['steps'][:7]) / 1000,  # în mii
    'Water': np.mean(health_metrics['water_liters'][:7]),
    'Mood': np.mean(health_metrics['mood_score'][:7])
}
week2_avg = {
    'Sleep': np.mean(health_metrics['sleep_hours'][7:]),
    'Steps': np.mean(health_metrics['steps'][7:]) / 1000,
    'Water': np.mean(health_metrics['water_liters'][7:]),
    'Mood': np.mean(health_metrics['mood_score'][7:])
}

metrics = list(week1_avg.keys())
fig.add_trace(go.Bar(x=metrics, y=list(week1_avg.values()),
                    name='Săptămâna 1', marker_color='lightblue'), row=3, col=2)
fig.add_trace(go.Bar(x=metrics, y=list(week2_avg.values()),
                    name='Săptămâna 2', marker_color='darkblue'), row=3, col=2)

# Layout pentru dashboard profesional
fig.update_layout(
    title_text="🏥 Personal Health Intelligence Dashboard",
    title_x=0.5,
    height=1000,
    showlegend=True,
    hovermode='closest'
)

# Update axes pentru claritate
fig.update_yaxes(title_text="Pași", row=2, col=1)
fig.update_xaxes(title_text="Data", row=2, col=1)
fig.update_xaxes(title_text="Ore Somn", row=1, col=2)
fig.update_yaxes(title_text="Nivel Energie", row=1, col=2)
fig.update_xaxes(title_text="Apă (L)", row=3, col=1)
fig.update_yaxes(title_text="Energie", row=3, col=1)

fig.show()

# 🎯 REZULTAT: Dashboard de sănătate complet care revelează:
# - Pattern-uri de somn cu vizualizare tip heatmap
# - Corelații clare între somn și energie
# - Timeline activitate zilnică vs. target-uri
# - Relația dintre hidratare și starea de spirit
# - Comparații săptămânale pentru progress tracking
#
# 💡 INSIGHT MAGIC: Interactivitatea revelează corelații ascunse!
# Hover pe orice punct pentru context complet

# 🔬 BONUS: Calculează corelația automată
correlation_sleep_energy = np.corrcoef(health_metrics['sleep_hours'], 
                                      health_metrics['energy_level'])[0,1]
print(f"🔍 Corelația Somn-Energie: {correlation_sleep_energy:.3f}")
print("💡 Insight: Somn de calitate = Energie ridicată!")

# Salvează dashboard-ul pentru tracking continuu
fig.write_html("health_dashboard.html")
print("✅ Dashboard salvat ca 'health_dashboard.html'")