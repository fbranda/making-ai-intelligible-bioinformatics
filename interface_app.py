"""Minimal Gradio recipe (Practice 6): calibrated probability + conformal set + top k-mer contributions.
Run `python case_study.py` first (it writes model.joblib), then `pip install gradio shap matplotlib` and `python interface_app.py`.
Privacy: inputs are processed in memory and never written to disk or logged."""

import joblib
import numpy as np
import shap
import gradio as gr

# Usa il backend 'Agg' per evitare crash di Matplotlib in ambienti web/server
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- CONFIGURAZIONE E CARICAMENTO MODELLO ---
REQUIRED_KEYS = {"gbm", "iso", "q", "kmers", "k"}

art = joblib.load("model.joblib")
if not (isinstance(art, dict) and REQUIRED_KEYS <= art.keys()):
    raise SystemExit(
        f"model.joblib contains {type(art).__name__}, not the expected artifact dict "
        f"{sorted(REQUIRED_KEYS)}. Re-run case_study.py to regenerate it."
    )

gbm = art["gbm"]
iso = art["iso"]
q = float(art["q"])
KMERS = art["kmers"]
K = int(art["k"])
KIDX = {kmer: i for i, kmer in enumerate(KMERS)}
explainer = shap.TreeExplainer(gbm)


# --- FUNZIONI DI SUPPORTO ---
def featurize(seq):
    """Converte una sequenza DNA in un vettore di conteggi di 4-mer."""
    x = np.zeros((1, len(KMERS)), dtype=np.float32)
    for i in range(len(seq) - K + 1):
        j = KIDX.get(seq[i:i + K])
        if j is not None:
            x[0, j] += 1
    return x


def predict(seq):
    """Gestisce l'input, calcola predizione, conformal set, SHAP e genera il grafico."""
    # Pulizia della sequenza
    seq = "".join(c for c in seq.upper() if c in "ACGT")
    if len(seq) < 50:
        return "Please enter at least 50 bp of A/C/G/T.", "", "", None

    # Feature extraction e predizione
    x = featurize(seq)
    p_raw = float(gbm.predict_proba(x)[0, 1])
    p_cal = float(iso.predict([p_raw])[0])

    # Split-conformal set at level (1 - ALPHA)
    label_set = [
        lab
        for lab, s in (("not bound", p_raw), ("bound", 1.0 - p_raw))
        if s <= q
    ]

    # SHAP values (gestisce sia lista che array a seconda della versione di SHAP)
    sv = explainer.shap_values(x, check_additivity=False)
    if isinstance(sv, list):
        sv = sv[1]                      # positive class
    phi = np.asarray(sv).reshape(-1)    # shape (n_features,)

    # Estrazione Top-5 k-mer
    top_idx = np.argsort(-np.abs(phi))[:5]
    
    # Testo dei top k-mer
    expl = "\n".join(
        f"{KMERS[i]}: {phi[i]:+.3f} (log-odds contribution)" for i in top_idx
    )

    # --- GENERAZIONE DELLA FIGURA (GRAFICO OTTIMIZZATO) ---
    top_kmers = [KMERS[i] for i in top_idx]
    top_vals = [phi[i] for i in top_idx]
    
    # Invertiamo l'ordine per far apparire il k-mer più importante in alto
    top_kmers.reverse()
    top_vals.reverse()
    
    # Colori: rosso per contributo positivo (bound), blu per negativo (not bound)
    colors = ['#d62728' if v > 0 else '#1f77b4' for v in top_vals]

    # Stile globale del grafico
    plt.rcParams.update({'font.size': 11, 'font.family': 'sans-serif'})
    
    # Altezza aumentata a 4 per dare respiro al grafico
    fig, ax = plt.subplots(figsize=(6.5, 4), dpi=150)
    
    ax.barh(top_kmers, top_vals, color=colors, height=0.6)
    
    # Rimuove i bordi superiore e destro per un look più pulito
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')
    
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel("SHAP value (log-odds impact)", fontweight='bold', color='#333333')
    
    # Titolo allineato a sinistra
    ax.set_title("Top-5 k-mer contributions", loc='left', fontweight='bold', color='#333333', pad=15)
    
    # Griglia leggera sull'asse x
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    
    # CORREZIONE: lascia il 10% di spazio in alto per non tagliare il titolo
    fig.tight_layout(rect=[0, 0, 1, 0.90])

    # Nota corretta (rimosso "Perspective")
    note = (
        "Attributions are correlational statements about this model, not causal claims; "
        "see Practice 3 of the main text. Model trained on simulated data only."
    )

    pred_text = (
        f"Calibrated P(bound) = {p_cal:.2f} (raw {p_raw:.2f}); "
        f"90% conformal set: {label_set or ['(empty)']}"
    )

    return pred_text, expl, note, fig


# --- INTERFACCIA GRADIO (LAYOUT A DUE COLONNE) ---
with gr.Blocks(title="Illustrative TF-binding model") as demo:
    gr.Markdown("## Illustrative TF-binding model (simulated data)")
    
    with gr.Row():
        # Colonna Sinistra: Input, Bottoni e Caveat
        with gr.Column(scale=1):
            seq_input = gr.Textbox(
                lines=6, 
                label="DNA sequence (>= 50 bp)",
                placeholder="ATCGGCTAGCTAAGCTAGCTAGGCTAAGCTAGCTAGCTAGCTAGCTAGGCTAAGCTAGCTAGCTAGCTAGCTAGCTA"
            )
            with gr.Row():
                clear_btn = gr.Button("Clear")
                submit_btn = gr.Button("Submit", variant="primary")
            
            caveat_out = gr.Textbox(label="Caveat", interactive=False, lines=3)
        
        # Colonna Destra: Output testuali e Grafico
        with gr.Column(scale=1):
            pred_out = gr.Textbox(label="Prediction and uncertainty")
            text_out = gr.Textbox(label="Top-5 k-mer contributions (text)", lines=6)
            plot_out = gr.Plot(label="Top-5 k-mer contributions (plot)")

    # Collega i bottoni alle funzioni
    submit_btn.click(
        fn=predict, 
        inputs=seq_input, 
        outputs=[pred_out, text_out, caveat_out, plot_out]
    )
    
    # Funzione per pulire tutti i campi
    clear_btn.click(
        fn=lambda: ("", "", "", "", None),
        inputs=None,
        outputs=[seq_input, pred_out, text_out, caveat_out, plot_out]
    )

if __name__ == "__main__":
    demo.launch()