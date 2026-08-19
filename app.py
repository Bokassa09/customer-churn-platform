import dash
from dash import html, dcc, callback, Input, Output, State
import requests

API_URL = "https://customer-churn-platform-f0g1.onrender.com"
#API_URL = "http://127.0.0.1:8000"

app = dash.Dash(__name__)

# Style global
COLORS = {
    "primary":    "#2c3e50",
    "secondary":  "#3498db",
    "success":    "#2ecc71",
    "danger":     "#e74c3c",
    "warning":    "#f39c12",
    "light":      "#f8f9fa",
    "border":     "#dee2e6",
    "text":       "#495057"
}

STYLE_CARD = {
    "backgroundColor": "white",
    "borderRadius": "10px",
    "padding": "25px",
    "boxShadow": "0 2px 10px rgba(0,0,0,0.1)",
    "marginBottom": "20px"
}

STYLE_INPUT = {
    "width": "100%",
    "padding": "8px",
    "borderRadius": "5px",
    "border": f"1px solid {COLORS['border']}",
    "marginBottom": "15px",
    "fontSize": "14px"
}

STYLE_LABEL = {
    "fontWeight": "bold",
    "color": COLORS["text"],
    "marginBottom": "5px",
    "display": "block",
    "fontSize": "13px"
}

app.layout = html.Div([

    # Header
html.Div([
    html.H1("🏦 Customer Churn Intelligence Platform",
            style={"color": "white", "margin": "0",
                   "fontSize": "28px"}),
    html.P("Détectez et expliquez le risque de départ de vos clients bancaires",
           style={"color": "rgba(255,255,255,0.8)",
                  "margin": "5px 0 5px 0", "fontSize": "14px"}),
    html.P("CatBoost · MLflow · Kedro · SHAP · Groq · Dash",
           style={"color": "rgba(255,255,255,0.5)",
                  "margin": "0", "fontSize": "12px",
                  "letterSpacing": "2px"}),
                  html.Span("✅ Modèle en production sur Render",
          style={"color": "#2ecc71",
                 "fontSize": "11px",
                 "marginTop": "5px",
                 "display": "block"})
], style={
    "backgroundColor": COLORS["primary"],
    "padding": "25px 40px",
    "marginBottom": "20px"
}),

# KPIs statiques
html.Div([
    html.Div([
        html.H4("87.5%", style={"color": COLORS["secondary"],
                                 "margin": "0", "fontSize": "24px"}),
        html.P("ROC-AUC", style={"color": COLORS["text"],
                                  "margin": "0", "fontSize": "12px"})
    ], style={"textAlign": "center", "flex": "1"}),

    html.Div([
        html.H4("69.0%", style={"color": COLORS["secondary"],
                                 "margin": "0", "fontSize": "24px"}),
        html.P("Recall", style={"color": COLORS["text"],
                                 "margin": "0", "fontSize": "12px"})
    ], style={"textAlign": "center", "flex": "1"}),

    html.Div([
        html.H4("61.0%", style={"color": COLORS["secondary"],
                                 "margin": "0", "fontSize": "24px"}),
        html.P("F1 Score", style={"color": COLORS["text"],
                                   "margin": "0", "fontSize": "12px"})
    ], style={"textAlign": "center", "flex": "1"}),

    html.Div([
        html.H4("10 000", style={"color": COLORS["secondary"],
                                  "margin": "0", "fontSize": "24px"}),
        html.P("Clients dans le dataset", style={"color": COLORS["text"],
                                           "margin": "0", "fontSize": "12px"})
    ], style={"textAlign": "center", "flex": "1"}),

], style={
    "display": "flex",
    "backgroundColor": "white",
    "padding": "15px 40px",
    "marginBottom": "20px",
    "boxShadow": "0 2px 5px rgba(0,0,0,0.05)"
}),

html.Div([
    html.P("🏦 Problématique métier",
           style={"fontWeight": "bold", "color": COLORS["primary"],
                  "marginBottom": "5px", "fontSize": "13px"}),
    html.P("Une banque constate que des clients ferment leur compte chaque mois. "
           "Acquérir un nouveau client coûte 5 à 7 fois plus cher que d'en retenir un. "
           "Elle veut donc identifier à l'avance les clients à risque de départ (churn) "
           "pour cibler ses actions de rétention.",
           style={"color": COLORS["text"], "fontSize": "13px",
                  "fontStyle": "italic", "margin": "0"})
], style={
    "backgroundColor": "#eaf4fb",
    "padding": "15px 40px",
    "marginBottom": "20px",
    "borderLeft": f"4px solid {COLORS['secondary']}"
}),


    # Contenu principal
    html.Div([

        # Formulaire
        html.Div([
            html.H3("📋 Informations du client",
                    style={"color": COLORS["primary"],
                           "marginTop": "0"}),
            html.P("Renseignez les informations du client pour analyser son risque de churn.",
                   style={"color": COLORS["text"], "fontSize": "13px"}),

            # 2 colonnes
            html.Div([

                # Colonne 1
                html.Div([
                    html.Label("Credit Score", style=STYLE_LABEL),
                    dcc.Input(id="credit-score", type="number",
                              value=650, style=STYLE_INPUT),

                    html.Label("Âge", style=STYLE_LABEL),
                    dcc.Input(id="age", type="number",
                              value=55, style=STYLE_INPUT),

                    html.Label("Ancienneté (années)", style=STYLE_LABEL),
                    dcc.Input(id="tenure", type="number",
                              value=2, style=STYLE_INPUT),

                    html.Label("Solde du compte (€)", style=STYLE_LABEL),
                    dcc.Input(id="balance", type="number",
                              value=125000, style=STYLE_INPUT),

                    html.Label("Salaire estimé (€)", style=STYLE_LABEL),
                    dcc.Input(id="salary", type="number",
                              value=80000, style=STYLE_INPUT),

                ], style={"width": "48%"}),

                # Colonne 2
                html.Div([
                    html.Label("Géographie", style=STYLE_LABEL),
                    dcc.Dropdown(
                        id="geography",
                        options=[
                            {"label": "🇫🇷 France", "value": "France"},
                            {"label": "🇩🇪 Allemagne", "value": "Germany"},
                            {"label": "🇪🇸 Espagne", "value": "Spain"}
                        ],
                        value="Germany",
                        style={"marginBottom": "15px"}
                    ),

                    html.Label("Genre", style=STYLE_LABEL),
                    dcc.Dropdown(
                        id="gender",
                        options=[
                            {"label": "Homme", "value": "Male"},
                            {"label": "Femme", "value": "Female"}
                        ],
                        value="Female",
                        style={"marginBottom": "15px"}
                    ),

                    html.Label("Nombre de produits", style=STYLE_LABEL),
                    dcc.Slider(id="num-products", min=1, max=4,
                               step=1, value=1,
                               marks={i: str(i) for i in range(1, 5)}),

                    html.Br(),

                    html.Label("Carte de crédit", style=STYLE_LABEL),
                    dcc.RadioItems(
                        id="has-crcard",
                        options=[
                            {"label": " Oui", "value": 1},
                            {"label": " Non", "value": 0}
                        ],
                        value=1,
                        inline=True,
                        style={"marginBottom": "15px"}
                    ),

                    html.Label("Membre actif", style=STYLE_LABEL),
                    dcc.RadioItems(
                        id="is-active",
                        options=[
                            {"label": " Oui", "value": 1},
                            {"label": " Non", "value": 0}
                        ],
                        value=0,
                        inline=True,
                        style={"marginBottom": "15px"}
                    ),

                ], style={"width": "48%"}),

            ], style={"display": "flex",
                      "justifyContent": "space-between"}),

        html.P("🔍 Les explications SHAP sont générées en temps réel pour chaque client analysé.",
       style={"color": "#999",
              "fontSize": "12px",
              "fontStyle": "italic",
              "marginTop": "10px"}),

            # Bouton
            html.Button("🔍 Analyser le client",
                        id="btn-analyser",
                        n_clicks=0,
                        style={
                            "backgroundColor": COLORS["secondary"],
                            "color": "white",
                            "padding": "12px 30px",
                            "border": "none",
                            "borderRadius": "5px",
                            "cursor": "pointer",
                            "fontSize": "16px",
                            "width": "100%",
                            "marginTop": "10px"
                        }),

        ], style=STYLE_CARD),

        # Zone résultat
        html.Div(id="zone-resultat"),

    ], style={"maxWidth": "900px", "margin": "0 auto",
              "padding": "0 20px"}),

    # Footer
    html.Div([
        html.P("Développé par Omer Bokassa Boueke | ML Engineer & MLOps | 2026",
               style={"color": "rgba(255,255,255,0.7)",
                      "margin": "0", "fontSize": "13px"})
    ], style={
        "backgroundColor": COLORS["primary"],
        "padding": "20px",
        "textAlign": "center",
        "marginTop": "40px"
    })

], style={"backgroundColor": COLORS["light"],
          "minHeight": "100vh",
          "fontFamily": "Arial, sans-serif"})


@callback(
    Output("zone-resultat", "children"),
    Input("btn-analyser", "n_clicks"),
    State("credit-score", "value"),
    State("geography", "value"),
    State("gender", "value"),
    State("age", "value"),
    State("tenure", "value"),
    State("balance", "value"),
    State("num-products", "value"),
    State("has-crcard", "value"),
    State("is-active", "value"),
    State("salary", "value"),
    prevent_initial_call=True
)
def analyser_client(n_clicks, credit_score, geography, gender,
                    age, tenure, balance, num_products,
                    has_crcard, is_active, salary):

    client = {
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": float(balance),
        "NumOfProducts": num_products,
        "HasCrCard": has_crcard,
        "IsActiveMember": is_active,
        "EstimatedSalary": float(salary)
    }

    try:
        res_predict = requests.post(f"{API_URL}/predict", json=client)
        res_explain = requests.post(f"{API_URL}/explain", json=client)
        result = res_predict.json()
        shap_result = res_explain.json()
        # Appel /analyze
        res_analyze = requests.post(f"{API_URL}/analyze", json=client)
        analyse_llm = res_analyze.json()["analyse_llm"]

        est_churn = result["prediction"] == 1
        couleur_bg = "#fdecea" if est_churn else "#eafaf1"
        couleur_border = COLORS["danger"] if est_churn else COLORS["success"]
        emoji = "❌" if est_churn else "✅"

        return html.Div([
    # Résultat principal
    html.Div([
        html.H2(f"{emoji} {result['message']}",
                style={"color": couleur_border,
                       "marginTop": "0"}),
        html.H3(
            f"Probabilité de churn : {result['probabilite']}%"
            if result['prediction'] == 1
            else f"Probabilité de fidélité : {result['probabilite']}%",
            style={"color": COLORS["text"]}
        ),
    ], style={
        **STYLE_CARD,
        "borderLeft": f"5px solid {couleur_border}",
        "backgroundColor": couleur_bg
    }),

        
# Top 3 SHAP
html.Div([
    html.H3("🔍 Pourquoi cette prédiction ?",
            style={"color": COLORS["primary"], "marginTop": "0"}),
    html.P("Les 3 facteurs principaux qui influencent cette décision :",
           style={"color": COLORS["text"], "fontSize": "13px"}),
    html.Div([
        html.Div([
            html.Span("🔴 " if info["sens"] == "↑ churn" else "🟢 ",
                      style={"fontSize": "20px"}),
            html.Strong(feat),
            html.Br(),
            html.Span(info['explication'],
                      style={"color": COLORS["text"],
                             "fontSize": "14px",
                             "marginLeft": "30px"}),
            html.Span(f" ({info['shap']:+.2f})",
                      style={"color": "#999", "fontSize": "12px"})
        ], style={
            "padding": "12px",
            "marginBottom": "10px",
            "backgroundColor": "#fef9e7" if info["sens"] == "↑ churn" else "#eafaf1",
            "borderRadius": "5px",
            "borderLeft": f"4px solid {'#e74c3c' if info['sens'] == '↑ churn' else '#2ecc71'}"
        })
        for feat, info in shap_result["top3_facteurs"].items()
    ])
], style=STYLE_CARD),


# Graphique SHAP
html.Div([
    html.H3("📊 Impact de chaque facteur",
            style={"color": COLORS["primary"], "marginTop": "0"}),
    html.P("🔴 Barre rouge = augmente le risque de churn   |   🟢 Barre verte = diminue le risque",
       style={"color": COLORS["text"], "fontSize": "13px"}),
    dcc.Graph(
        figure={
            "data": [{
                "type": "bar",
                "orientation": "h",
                "x": list(shap_result["top3_facteurs"].values())[i]["shap"]
                     if False else
                     [info["shap"] for info in shap_result["top3_facteurs"].values()],
                "y": list(shap_result["top3_facteurs"].keys()),
                "marker": {
                    "color": [
                        "#e74c3c" if info["shap"] > 0 else "#2ecc71"
                        for info in shap_result["top3_facteurs"].values()
                    ]
                },
                "type": "bar",
                "orientation": "h"
            }],
            "layout": {
                "title": "Top 3 facteurs SHAP",
                "xaxis": {"title": "Valeur SHAP"},
                "yaxis": {"title": ""},
                "plot_bgcolor": "white",
                "paper_bgcolor": "white",
                "height": 250,
                "margin": {"l": 150, "r": 20, "t": 40, "b": 40}
            }
        },
        config={"displayModeBar": False}
    )
], style=STYLE_CARD),

# Analyse LLM
html.Div([
    html.H3("🤖 Analyse IA : Recommandation conseiller",
            style={"color": COLORS["primary"], "marginTop": "0"}),
    html.P("Généré par Groq Compound Mini",
           style={"color": "#999", "fontSize": "11px",
                  "marginBottom": "10px"}),
    html.P(analyse_llm,
           style={"color": COLORS["text"],
                  "fontSize": "14px",
                  "lineHeight": "1.8",
                  "fontStyle": "italic",
                  "borderLeft": f"3px solid {COLORS['secondary']}",
                  "paddingLeft": "15px"})
], style=STYLE_CARD)

        ])


    

    except Exception as e:
        return html.Div([
            html.H3("⚠️ Erreur de connexion à l'API",
                    style={"color": COLORS["warning"]}),
            html.P(f"Détail : {str(e)}",
                   style={"color": COLORS["text"]})
        ], style=STYLE_CARD)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)