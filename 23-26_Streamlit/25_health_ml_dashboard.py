import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🤖 AI Health Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'health_data' not in st.session_state:
    st.session_state.health_data = pd.DataFrame()
if 'energy_model' not in st.session_state:
    st.session_state.energy_model = None
if 'model_accuracy' not in st.session_state:
    st.session_state.model_accuracy = None

# Custom styling
st.markdown("""
<style>
.ai-metric {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.5rem;
    border-radius: 0.75rem;
    color: white;
    text-align: center;
    box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
}
.prediction-card {
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    padding: 1.5rem;
    border-radius: 0.75rem;
    color: white;
    margin: 1rem 0;
}
.ai-insight {
    background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
    padding: 1rem;
    border-radius: 0.5rem;
    color: white;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# Header
st.title("🤖 AI-Powered Health Dashboard")
st.markdown("*Machine Learning meets Personal Wellness*")

# Sidebar for data input and ML controls
with st.sidebar:
    st.header("📝 Health Data Input")
    
    # Data input form (similar to before but more ML-focused)
    log_date = st.date_input("Date", datetime.now().date())
    
    col1, col2 = st.columns(2)
    with col1:
        sleep_hours = st.slider("Sleep Hours", 0.0, 12.0, 7.5, 0.5)
        sleep_quality = st.slider("Sleep Quality", 1, 10, 7)
        exercise_minutes = st.slider("Exercise (min)", 0, 180, 30)
    
    with col2:
        mood_score = st.slider("Mood", 1, 10, 7)
        stress_level = st.slider("Stress", 1, 10, 4)
        energy_level = st.slider("Energy", 1, 10, 7)
    
    # Advanced features for ML
    st.subheader("🔬 Advanced Metrics")
    caffeine_cups = st.slider("Caffeine (cups)", 0, 8, 2)
    screen_time = st.slider("Screen Time (hours)", 0, 16, 6)
    outdoor_time = st.slider("Outdoor Time (min)", 0, 240, 30)
    
    if st.button("💾 Add Data Point", type="primary"):
        new_data = {
            'date': log_date,
            'sleep_hours': sleep_hours,
            'sleep_quality': sleep_quality,
            'exercise_minutes': exercise_minutes,
            'mood_score': mood_score,
            'stress_level': stress_level,
            'energy_level': energy_level,
            'caffeine_cups': caffeine_cups,
            'screen_time': screen_time,
            'outdoor_time': outdoor_time,
            'day_of_week': log_date.weekday(),  # 0=Monday, 6=Sunday
            'is_weekend': log_date.weekday() >= 5
        }
        
        st.session_state.health_data = pd.concat([
            st.session_state.health_data,
            pd.DataFrame([new_data])
        ], ignore_index=True)
        
        st.success("✅ Data added!")
        st.rerun()
    
    st.markdown("---")
    
    # ML Model Controls
    st.header("🤖 AI Model")
    
    if len(st.session_state.health_data) >= 10:
        if st.button("🚀 Train Energy Predictor", type="secondary"):
            # Train machine learning model
            df = st.session_state.health_data.copy()
            
            # Feature engineering
            features = ['sleep_hours', 'sleep_quality', 'exercise_minutes', 
                       'mood_score', 'stress_level', 'caffeine_cups', 
                       'screen_time', 'outdoor_time', 'day_of_week']
            
            X = df[features]
            y = df['energy_level']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test)
            accuracy = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            # Save model
            st.session_state.energy_model = model
            st.session_state.model_accuracy = {
                'r2': accuracy,
                'mae': mae,
                'feature_importance': dict(zip(features, model.feature_importances_))
            }
            
            st.success(f"✅ Model trained! R² = {accuracy:.3f}")
            st.rerun()
    
    else:
        st.info(f"Need {10 - len(st.session_state.health_data)} more data points to train AI model")

# Main dashboard
if len(st.session_state.health_data) > 0:
    df = st.session_state.health_data.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # AI Predictions Section
    if st.session_state.energy_model is not None:
        st.markdown("""
        <div class="prediction-card">
            <h3>🔮 AI Predictions & Recommendations</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Create prediction interface
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("🎯 Predict Tomorrow's Energy")
            pred_sleep = st.slider("Planned Sleep", 6.0, 10.0, 8.0, key="pred_sleep")
            pred_exercise = st.slider("Planned Exercise", 0, 120, 30, key="pred_exercise")
            pred_caffeine = st.slider("Planned Caffeine", 0, 5, 2, key="pred_caffeine")
            
            # Make prediction
            tomorrow = datetime.now().date() + timedelta(days=1)
            features = np.array([[
                pred_sleep, 8.0, pred_exercise, 7.0, 4.0,  # sleep_hours, sleep_quality, exercise, mood, stress
                pred_caffeine, 6.0, 30, tomorrow.weekday(), tomorrow.weekday() >= 5
            ]])
            
            predicted_energy = st.session_state.energy_model.predict(features)[0]
            
            st.markdown(f"""
            <div class="ai-metric">
                <h4>Predicted Energy</h4>
                <h2>{predicted_energy:.1f}/10</h2>
                <p>{"🚀 High Energy!" if predicted_energy >= 8 else "⚡ Good Energy" if predicted_energy >= 6 else "💤 Low Energy"}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.subheader("📊 Model Performance")
            accuracy_data = st.session_state.model_accuracy
            
            st.markdown(f"""
            <div class="ai-metric">
                <h4>Model Accuracy</h4>
                <h2>{accuracy_data['r2']:.1%}</h2>
                <p>R² Score</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="ai-insight">
                <strong>Mean Error:</strong> {accuracy_data['mae']:.2f} energy points
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.subheader("🎯 Feature Importance")
            importance_df = pd.DataFrame([
                {'feature': k, 'importance': v} 
                for k, v in accuracy_data['feature_importance'].items()
            ]).sort_values('importance', ascending=True)
            
            fig_importance = px.bar(
                importance_df, 
                x='importance', 
                y='feature',
                orientation='h',
                title="What Affects Your Energy Most?"
            )
            fig_importance.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig_importance, use_container_width=True)
        
        # Smart recommendations
        st.subheader("🧠 AI Recommendations")
        top_factors = sorted(accuracy_data['feature_importance'].items(), 
                           key=lambda x: x[1], reverse=True)[:3]
        
        recommendations = []
        for factor, importance in top_factors:
            if factor == 'sleep_hours':
                recommendations.append(f"💤 **Sleep is {importance:.1%} of your energy!** Aim for 7.5-8.5 hours nightly.")
            elif factor == 'exercise_minutes':
                recommendations.append(f"🏃‍♀️ **Exercise drives {importance:.1%} of energy levels.** Even 20 minutes helps!")
            elif factor == 'stress_level':
                recommendations.append(f"😤 **Stress impacts {importance:.1%} of your energy.** Try meditation or breathing exercises.")
        
        for rec in recommendations:
            st.markdown(f"""
            <div class="ai-insight">
                {rec}
            </div>
            """, unsafe_allow_html=True)
    
    # Enhanced visualizations with ML insights
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["📈 Trends", "🤖 AI Analysis", "🔬 Experiments"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Energy prediction vs actual (if model exists)
            if st.session_state.energy_model is not None and len(df) > 5:
                # Make predictions for historical data
                features = ['sleep_hours', 'sleep_quality', 'exercise_minutes', 
                           'mood_score', 'stress_level', 'caffeine_cups', 
                           'screen_time', 'outdoor_time', 'day_of_week']
                
                X_historical = df[features]
                predicted_energy = st.session_state.energy_model.predict(X_historical)
                
                fig_pred = go.Figure()
                fig_pred.add_trace(go.Scatter(
                    x=df['date'], y=df['energy_level'],
                    mode='markers', name='Actual Energy',
                    marker=dict(color='blue', size=8)
                ))
                fig_pred.add_trace(go.Scatter(
                    x=df['date'], y=predicted_energy,
                    mode='lines', name='AI Prediction',
                    line=dict(color='red', width=2)
                ))
                fig_pred.update_layout(title="AI Predictions vs Reality", height=400)
                st.plotly_chart(fig_pred, use_container_width=True)
            else:
                # Regular energy trend
                fig_energy = px.line(df, x='date', y='energy_level', 
                                   title="Energy Levels Over Time")
                st.plotly_chart(fig_energy, use_container_width=True)
        
        with col2:
            # Multi-dimensional correlation heatmap
            numeric_cols = ['sleep_hours', 'sleep_quality', 'exercise_minutes', 
                           'mood_score', 'stress_level', 'energy_level']
            if all(col in df.columns for col in numeric_cols):
                corr_matrix = df[numeric_cols].corr()
                
                fig_heatmap = px.imshow(
                    corr_matrix,
                    title="Health Metrics Correlation Matrix",
                    color_continuous_scale='RdBu'
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)
    
    with tab2:
        if st.session_state.energy_model is not None:
            st.subheader("🤖 Advanced AI Analysis")
            
            # Optimal scenarios finder
            st.markdown("### 🎯 Find Your Optimal Conditions")
            
            # Find days with highest energy
            top_energy_days = df.nlargest(5, 'energy_level')
            avg_conditions = top_energy_days[['sleep_hours', 'exercise_minutes', 
                                            'caffeine_cups', 'stress_level']].mean()
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Your Peak Energy Formula:**")
                for metric, value in avg_conditions.items():
                    st.write(f"• {metric.replace('_', ' ').title()}: {value:.1f}")
            
            with col2:
                # What-if analysis
                st.markdown("**What-If Analysis:**")
                if st.button("🧪 Test +1 Hour Sleep"):
                    test_features = df[features].iloc[-1:].copy()
                    test_features['sleep_hours'] += 1
                    prediction = st.session_state.energy_model.predict(test_features)[0]
                    current = df['energy_level'].iloc[-1]
                    change = prediction - current
                    st.write(f"Predicted energy change: {change:+.1f} points")
        
        else:
            st.info("🤖 Train the AI model to unlock advanced analysis!")
    
    with tab3:
        st.subheader("🔬 Personal Health Experiments")
        
        # Experiment tracker
        st.markdown("### Design Your Own Health Experiments")
        
        experiment_type = st.selectbox("Experiment Type", [
            "Sleep Duration Impact",
            "Exercise Timing Effect", 
            "Caffeine Optimization",
            "Stress Management"
        ])
        
        if experiment_type == "Sleep Duration Impact":
            st.markdown("""
            **Experiment:** Track energy levels for different sleep durations
            
            **Protocol:**
            - Week 1: Sleep 7 hours nightly
            - Week 2: Sleep 8 hours nightly  
            - Week 3: Sleep 9 hours nightly
            - Compare average energy levels
            """)
        
        # Add experiment results if data exists
        if len(df) >= 21:  # 3 weeks of data
            recent_data = df.tail(21)
            weekly_energy = recent_data.groupby(recent_data.index // 7)['energy_level'].mean()
            
            if len(weekly_energy) >= 3:
                st.write("**Your Last 3 Weeks:**")
                for i, energy in enumerate(weekly_energy):
                    st.write(f"Week {i+1}: {energy:.1f} average energy")

else:
    st.info("👆 Start by adding your health data to unlock AI predictions!")
    
    # Load comprehensive demo data
    if st.button("🎯 Load AI Demo Data (30 days)"):
        dates = [datetime.now().date() - timedelta(days=i) for i in range(30)]
        demo_data = []
        
        for date in dates:
            # Create realistic patterns
            is_weekend = date.weekday() >= 5
            base_sleep = 7.5 + (0.5 if is_weekend else 0) + np.random.normal(0, 0.5)
            base_exercise = 30 + (10 if is_weekend else 0) + np.random.normal(0, 15)
            
            demo_data.append({
                'date': date,
                'sleep_hours': max(5, min(10, base_sleep)),
                'sleep_quality': np.random.randint(6, 10),
                'exercise_minutes': max(0, base_exercise),
                'mood_score': np.random.randint(6, 10),
                'stress_level': np.random.randint(2, 8),
                'energy_level': max(3, min(10, base_sleep * 0.8 + base_exercise * 0.02 + np.random.normal(0, 1))),
                'caffeine_cups': np.random.randint(1, 4),
                'screen_time': np.random.randint(4, 12),
                'outdoor_time': np.random.randint(10, 120),
                'day_of_week': date.weekday(),
                'is_weekend': is_weekend
            })
        
        st.session_state.health_data = pd.DataFrame(demo_data)
        st.success("AI demo data loaded! Now train your model to see predictions.")
        st.rerun()

# Footer
st.markdown("---")
st.markdown("*AI-Powered Health Tracking • The future of personal wellness 🤖*")