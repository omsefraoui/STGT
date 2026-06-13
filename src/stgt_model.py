#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STGT Ultimate - Spatio-Temporal Graph Transformer (Version Optimale - CORRIGÉE)

Combine les points forts de:
- STGT_corrected.py: Code propre, modulaire, CLI, production-ready
- Spatio-Temporal_GAN.py: Visualisations professionnelles, analyse complète

Architecture STGT:
- Temporal Transformer (4 layers, 8 heads)
- Spatial GAT (3 layers, 4 heads)
- Spatio-Temporal Fusion (cross-attention)
- Prediction Head (Dense layers)

FIX: Correction bug dimension dans GraphAttentionLayer (ligne 351)

Auteur: Omar, ENSAO - Université Mohammed Premier
Date: Janvier 2026
Version: 1.1 (Corrigée)
"""

import argparse
import json
import pickle
import warnings
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Dict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             mean_absolute_percentage_error, r2_score)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers, callbacks

warnings.filterwarnings("ignore")

# Configuration matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ─── Polices agrandies pour publication ───
#plt.rcParams.update({
 #   'axes.titlesize':   34,
  #  'axes.titleweight': 'bold',
   # 'axes.labelsize':   30,
    #'axes.labelweight': 'bold',
    #'xtick.labelsize':  26,
    #'ytick.labelsize':  26,
    #'legend.fontsize':  24,
    #'figure.titlesize': 24,
    #'font.size':        14,
#})


# ─── Polices très grandes pour conférence ───
plt.rcParams.update({

    # titre principal figure
    'figure.titlesize': 34,
    'figure.titleweight': 'bold',

    # titres des subplots
    'axes.titlesize': 30,
    'axes.titleweight': 'bold',

    # labels axes
    'axes.labelsize': 30,
    'axes.labelweight': 'bold',

    # ticks axes
    'xtick.labelsize': 26,
    'ytick.labelsize': 26,

    # légende
    'legend.fontsize': 22,

    # police générale
    'font.size': 24,

    # largeur lignes
    'lines.linewidth': 6,

    # taille marqueurs
    'lines.markersize': 14,

    # taille figure par défaut
    'figure.figsize': (12, 8),

    # résolution publication
    'figure.dpi': 300
})

# ============================================================================
# CONFIGURATION ET REPRODUCTIBILITÉ
# ============================================================================

class Config:
    """Configuration globale du modèle et de l'entraînement."""

    # Données
    DATA_DIR = "data"
    WINDOW_SIZE = 12
    HORIZON = 1

    # Architecture
    D_MODEL = 128
    TEMPORAL_HEADS = 8
    TEMPORAL_LAYERS = 4
    DFF = 512
    SPATIAL_HEADS = 4
    SPATIAL_LAYERS = 3
    FUSION_HEADS = 4
    DROPOUT = 0.1
    USE_FUSION = True

    # Entraînement
    EPOCHS = 100
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0001
    TRAIN_RATIO = 0.70
    VAL_RATIO = 0.15

    # Callbacks
    EARLY_STOPPING_PATIENCE = 20
    REDUCE_LR_PATIENCE = 10
    REDUCE_LR_FACTOR = 0.5
    MIN_LR = 1e-7

    # Sorties
    MODEL_DIR = Path("models")
    FIGURE_DIR = Path("figures")

    # Reproductibilité
    SEED = 42


def set_seed(seed: int = 42):
    """Configure la reproductibilité."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print(f"✓ Seed fixé: {seed}")


# ============================================================================
# CHARGEMENT ET PRÉPARATION DONNÉES
# ============================================================================

class DataLoader:
    """Gestionnaire de chargement et préparation des données."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)

    def load_all(self) -> Dict:
        """
        Charge toutes les données nécessaires.

        Returns:
            Dict contenant df, coords, matrices, réseau
        """
        print("\n" + "=" * 80)
        print("📂 CHARGEMENT DES DONNÉES")
        print("=" * 80)

        # Données temporelles
        print("\n📊 Chargement spatiotemporal_data.csv...")
        df = pd.read_csv(self.data_dir / 'spatiotemporal_data.csv',
                         parse_dates=['date'])

        # Coordonnées GPS
        print("📍 Chargement road_coordinates.csv...")
        coords_df = pd.read_csv(self.data_dir / 'road_coordinates.csv')
        coords = list(zip(coords_df['latitude'], coords_df['longitude']))

        # Matrices
        print("🗺️  Chargement matrices...")
        distance_matrix = np.load(self.data_dir / 'distance_matrix.npy')
        A_distance = np.load(self.data_dir / 'adjacency_distance.npy')
        A_similarity = np.load(self.data_dir / 'adjacency_similarity.npy')
        A_topology = np.load(self.data_dir / 'adjacency_topology.npy')

        # Graphe réseau
        print("🔗 Chargement road_network.pkl...")
        with open(self.data_dir / 'road_network.pkl', 'rb') as f:
            road_network = pickle.load(f)

        print(f"\n✓ Données chargées!")
        print(f"  - Observations: {len(df):,}")
        print(f"  - Segments: {df['road_id'].nunique()}")
        print(f"  - Période: {df['date'].min()} à {df['date'].max()}")

        return {
            'df': df,
            'coords': coords,
            'distance_matrix': distance_matrix,
            'A_distance': A_distance,
            'A_similarity': A_similarity,
            'A_topology': A_topology,
            'road_network': road_network
        }


class DataPreprocessor:
    """Préparation des séquences spatio-temporelles."""

    def __init__(self, window_size: int = 12, horizon: int = 1):
        self.window_size = window_size
        self.horizon = horizon

    def create_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Crée séquences [samples, roads, window, features].

        Returns:
            X, y, feature_cols
        """
        print("\n" + "=" * 80)
        print("🔧 CRÉATION SÉQUENCES SPATIO-TEMPORELLES")
        print("=" * 80)

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["date", "road_index"]).reset_index(drop=True)

        # Features numériques
        feature_cols = [
            "traffic_vpj", "heavy_vehicle_pct", "age_years", "thickness_cm",
            "IRI", "PCI", "degree_centrality",
            "IRI_rolling_mean_3m", "IRI_rolling_std_3m",
            "IRI_lag_1m", "IRI_lag_3m", "IRI_delta",
            "month", "quarter",
        ]

        # Encodage catégoriel
        for col, new_col in [
            ("road_type", "road_type_enc"),
            ("climate_zone", "climate_zone_enc"),
            ("pavement_type", "pavement_type_enc"),
        ]:
            if col in df.columns:
                le = LabelEncoder()
                df[new_col] = le.fit_transform(df[col].astype(str))
                feature_cols.append(new_col)

        # Gestion NaN
        df[feature_cols] = df[feature_cols].ffill().bfill()

        roads = sorted(df["road_index"].unique())
        timestamps = sorted(df["date"].unique())
        n_roads, n_time = len(roads), len(timestamps)

        print(f"\n📊 Dimensions:")
        print(f"  - Segments: {n_roads}")
        print(f"  - Timestamps: {n_time}")
        print(f"  - Features: {len(feature_cols)}")

        # Matrice [time, roads, features]
        feat = np.zeros((n_time, n_roads, len(feature_cols)), dtype=np.float32)
        target = np.zeros((n_time, n_roads), dtype=np.float32)

        road_to_pos = {r: i for i, r in enumerate(roads)}
        time_to_pos = {t: i for i, t in enumerate(timestamps)}

        for _, row in df.iterrows():
            ti = time_to_pos[row["date"]]
            ri = road_to_pos[row["road_index"]]
            feat[ti, ri, :] = row[feature_cols].values
            target[ti, ri] = row["IRI"]

        # Fenêtres glissantes
        X_list, y_list = [], []
        for start in range(n_time - self.window_size - self.horizon + 1):
            end = start + self.window_size
            y_t = end + self.horizon - 1
            X_list.append(feat[start:end, :, :])
            y_list.append(target[y_t, :])

        X = np.transpose(np.stack(X_list), (0, 2, 1, 3))
        y = np.stack(y_list)[:, :, None].astype(np.float32)

        print(f"✓ Séquences créées: X{X.shape}, y{y.shape}")
        return X, y, feature_cols

    def split_temporal(self, X: np.ndarray, y: np.ndarray,
                       train_ratio: float = 0.70, val_ratio: float = 0.15):
        """Split chronologique."""
        n = X.shape[0]
        n_train = int(train_ratio * n)
        n_val = int(val_ratio * n)

        splits = {
            'train': (X[:n_train], y[:n_train]),
            'val': (X[n_train:n_train + n_val], y[n_train:n_train + n_val]),
            'test': (X[n_train + n_val:], y[n_train + n_val:])
        }

        print(f"\n📊 Split temporel:")
        for name, (x, _) in splits.items():
            print(f"  - {name.capitalize()}: {x.shape[0]} samples")

        return splits

    def standardize(self, train, val, test):
        """Standardisation fit sur train."""
        print("\n🔧 Standardisation...")

        X_train, y_train = train
        X_val, y_val = val
        X_test, y_test = test

        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        # Flatten et transform
        def transform_X(X, fit=False):
            X_2d = X.reshape(-1, X.shape[-1])
            if fit:
                X_2d = scaler_X.fit_transform(X_2d)
            else:
                X_2d = scaler_X.transform(X_2d)
            return X_2d.reshape(X.shape).astype(np.float32)

        def transform_y(y, fit=False):
            if fit:
                return scaler_y.fit_transform(y.reshape(-1, 1)).reshape(y.shape).astype(np.float32)
            else:
                return scaler_y.transform(y.reshape(-1, 1)).reshape(y.shape).astype(np.float32)

        return {
            'train': (transform_X(X_train, fit=True), transform_y(y_train, fit=True)),
            'val': (transform_X(X_val), transform_y(y_val)),
            'test': (transform_X(X_test), transform_y(y_test)),
            'scalers': (scaler_X, scaler_y)
        }


# ============================================================================
# ARCHITECTURE MODÈLE
# ============================================================================

class PositionalEncoding(layers.Layer):
    """Encodage positionnel sinusoïdal."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = np.arange(max_len)[:, None]
        div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = np.zeros((max_len, d_model), dtype=np.float32)
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe = tf.constant(pe[None, :, :], dtype=tf.float32)

    def call(self, x):
        return x + self.pe[:, :tf.shape(x)[1], :]


class TransformerEncoderBlock(layers.Layer):
    """Transformer Encoder avec Multi-Head Attention."""

    def __init__(self, d_model: int, num_heads: int, dff: int, dropout: float):
        super().__init__()
        self.mha = layers.MultiHeadAttention(num_heads=num_heads,
                                             key_dim=d_model // num_heads)
        self.ffn = keras.Sequential([
            layers.Dense(dff, activation="relu"),
            layers.Dense(d_model),
        ])
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        self.drop1 = layers.Dropout(dropout)
        self.drop2 = layers.Dropout(dropout)

    def call(self, x, training=False):
        attn = self.mha(x, x, training=training)
        x = self.norm1(x + self.drop1(attn, training=training))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.drop2(ffn_out, training=training))


class GraphAttentionLayer(layers.Layer):
    """
    GAT multi-têtes avec masquage.
    FIX: Correction du bug de dimension dans le calcul de l'attention.
    """

    def __init__(self, units: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.units = units
        self.num_heads = num_heads
        self.W = [layers.Dense(units, use_bias=False) for _ in range(num_heads)]
        self.a = [layers.Dense(1, use_bias=False) for _ in range(num_heads)]
        self.leaky = layers.LeakyReLU(0.2)
        self.drop = layers.Dropout(dropout)
        # Projection de sortie (créée une seule fois pour compatibilité tf.function)
        self.out_proj = layers.Dense(units)

    def call(self, h, adjacency, training=False):
        """
        h: [batch, nodes, units]
        adjacency: [nodes, nodes]

        FIX: Utilisation correcte de broadcasting pour la concaténation.
        """
        A = tf.where(tf.cast(adjacency, tf.float32) > 0, 0.0, -1e9)

        outs = []
        for k in range(self.num_heads):
            Wh = self.W[k](h)  # [batch, nodes, units]

            # FIX: Broadcasting correct pour créer [batch, nodes, nodes, 2*units]
            num_nodes = tf.shape(Wh)[1]

            # Wh_i: [batch, nodes, 1, units] répété pour [batch, nodes, nodes, units]
            Wh_i = tf.expand_dims(Wh, 2)  # [batch, nodes, 1, units]
            Wh_i = tf.tile(Wh_i, [1, 1, num_nodes, 1])  # [batch, nodes, nodes, units]

            # Wh_j: [batch, 1, nodes, units] répété pour [batch, nodes, nodes, units]
            Wh_j = tf.expand_dims(Wh, 1)  # [batch, 1, nodes, units]
            Wh_j = tf.tile(Wh_j, [1, num_nodes, 1, 1])  # [batch, nodes, nodes, units]

            # Concaténation: [batch, nodes, nodes, 2*units]
            combined = tf.concat([Wh_i, Wh_j], axis=-1)

            # Calcul attention
            e = self.leaky(self.a[k](combined))  # [batch, nodes, nodes, 1]
            e = tf.squeeze(e, axis=-1)  # [batch, nodes, nodes]

            # Masquage + softmax
            alpha = tf.nn.softmax(e + A, axis=-1)
            alpha = self.drop(alpha, training=training)

            # Agrégation
            out = tf.matmul(alpha, Wh)  # [batch, nodes, units]
            outs.append(out)

        # Concaténer toutes les têtes et projeter
        h_out = tf.concat(outs, axis=-1)  # [batch, nodes, num_heads*units]
        h_out = self.out_proj(h_out)  # [batch, nodes, units]
        return h_out


class SpatioTemporalFusion(layers.Layer):
    """Cross-attention bidirectionnelle."""

    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.t2s = layers.MultiHeadAttention(num_heads=num_heads,
                                             key_dim=d_model // num_heads)
        self.s2t = layers.MultiHeadAttention(num_heads=num_heads,
                                             key_dim=d_model // num_heads)
        self.gate = layers.Dense(d_model, activation="sigmoid")
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        self.drop = layers.Dropout(dropout)

    def call(self, temporal_emb, spatial_emb, training=False):
        t2s_out = self.t2s(spatial_emb, temporal_emb, training=training)
        s2t_out = self.s2t(temporal_emb, spatial_emb, training=training)

        fused1 = self.norm1(temporal_emb + self.drop(s2t_out, training=training))
        fused2 = self.norm2(spatial_emb + self.drop(t2s_out, training=training))

        gate = self.gate(tf.concat([fused1, fused2], axis=-1))
        return gate * fused1 + (1 - gate) * fused2


class STGTModel(keras.Model):
    """Modèle STGT complet."""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.embedding = layers.Dense(config.D_MODEL)
        self.pos_enc = PositionalEncoding(config.D_MODEL)

        self.temporal_blocks = [
            TransformerEncoderBlock(config.D_MODEL, config.TEMPORAL_HEADS,
                                    config.DFF, config.DROPOUT)
            for _ in range(config.TEMPORAL_LAYERS)
        ]

        self.spatial_blocks = [
            GraphAttentionLayer(config.D_MODEL, config.SPATIAL_HEADS,
                                config.DROPOUT)
            for _ in range(config.SPATIAL_LAYERS)
        ]

        self.fusion = SpatioTemporalFusion(config.D_MODEL, config.FUSION_HEADS,
                                           config.DROPOUT)

        self.pred_head = keras.Sequential([
            layers.Dense(64, activation="relu"),
            layers.Dropout(config.DROPOUT),
            layers.Dense(32, activation="relu"),
            layers.Dense(1),
        ])

    def call(self, inputs, adjacency, training=False):
        # Itération sûre en mode graph: décomposer les routes (axis=1)
        roads_seq = tf.unstack(inputs, axis=1)  # liste de tenseurs [batch, window, features]

        # Temporal encoding
        temporal_outputs = []
        for road_seq in roads_seq:
            x = self.embedding(road_seq)
            x = self.pos_enc(x)
            for block in self.temporal_blocks:
                x = block(x, training=training)
            temporal_outputs.append(tf.reduce_mean(x, axis=1))

        temporal_emb = tf.stack(temporal_outputs, axis=1)

        # Spatial encoding
        spatial_emb = temporal_emb
        for block in self.spatial_blocks:
            spatial_emb = block(spatial_emb, adjacency, training=training)

        # Fusion
        if self.config.USE_FUSION:
            fused = self.fusion(temporal_emb, spatial_emb, training=training)
        else:
            fused = spatial_emb

        return self.pred_head(fused)


class STGTWrapper(keras.Model):
    """Wrapper pour model.fit()."""

    def __init__(self, stgt_model, adjacency_matrix):
        super().__init__()
        self.stgt = stgt_model
        self.adjacency = tf.constant(adjacency_matrix, dtype=tf.float32)

    def call(self, inputs, training=False):
        return self.stgt(inputs, self.adjacency, training=training)


# ============================================================================
# VISUALISATIONS
# ============================================================================

class Visualizer:
    """Gestionnaire de visualisations."""

    def __init__(self, output_dir: Path = Path("figures")):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)

    def plot_network(self, graph, coords, iri_values):
        """Figure 1: Réseau routier."""
        print("\n🎨 Génération network_visualization.eps...")

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        pos = {i: (c[1], c[0]) for i, c in enumerate(coords)}

        # Panel 1: Réseau
        colors = plt.cm.RdYlGn_r((iri_values - iri_values.min()) /
                                 (iri_values.max() - iri_values.min()))
        nx.draw_networkx_edges(graph, pos, alpha=0.2, width=0.5, ax=axes[0])
        nx.draw_networkx_nodes(graph, pos, node_color=colors,
                               node_size=100, ax=axes[0])
        axes[0].set_title('Road Network State', fontweight='bold')
        axes[0].set_xlabel('Longitude')
        axes[0].set_ylabel('Latitude')

        # Panel 2: Distribution
        axes[1].hist(iri_values, bins=30, alpha=0.7, edgecolor='black')
        axes[1].axvline(iri_values.mean(), color='red', linestyle='--',
                        label=f'Mean: {iri_values.mean():.2f}')
        axes[1].set_xlabel('IRI (mm/m)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('IRI Distribution', fontweight='bold')
        axes[1].legend()

        # Panel 3: Stats
        stats_text = (f'Network Statistics\n\n'
                      f'Segments: {len(graph.nodes)}\n'
                      f'Connections: {len(graph.edges)}\n'
                      f'Mean IRI: {iri_values.mean():.2f} mm/m\n'
                      f'Std IRI: {iri_values.std():.2f} mm/m')
        axes[2].text(0.5, 0.5, stats_text, ha='center', va='center',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[2].axis('off')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'network_visualization.eps',
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Sauvegardé")

    def plot_training(self, history_df):
        """Figure 2: Courbes d'entraînement."""
        print("\n📈 Génération training_curves.eps...")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        epochs = range(1, len(history_df) + 1)

        # Loss
        axes[0].plot(epochs, history_df['loss'], 'b-', label='Train', lw=2)
        axes[0].plot(epochs, history_df['val_loss'], 'r-', label='Val', lw=2)
        axes[0].set_xlabel('Epoch', fontweight='bold')
        axes[0].set_ylabel('Loss (MSE)', fontweight='bold')
        axes[0].set_title('Training Loss', fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # MAE
        axes[1].plot(epochs, history_df['mae'], 'b-', label='Train', lw=2)
        axes[1].plot(epochs, history_df['val_mae'], 'r-', label='Val', lw=2)
        axes[1].set_xlabel('Epoch', fontweight='bold')
        axes[1].set_ylabel('MAE (mm/m)', fontweight='bold')
        axes[1].set_title('Training MAE', fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_curves.eps',
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Sauvegardé")

    def plot_evaluation(self, y_true, y_pred):
        """Figure 3: Résultats évaluation."""
        print("\n📊 Génération evaluation_results.eps...")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        y_true_flat, y_pred_flat = y_true.flatten(), y_pred.flatten()
        errors = y_true_flat - y_pred_flat

        # Scatter
        axes[0, 0].scatter(y_true_flat, y_pred_flat, alpha=0.5, s=20)
        axes[0, 0].plot([y_true_flat.min(), y_true_flat.max()],
                        [y_true_flat.min(), y_true_flat.max()],
                        'r--', lw=2, label='Perfect')
        axes[0, 0].set_xlabel('Actual IRI', fontweight='bold')
        axes[0, 0].set_ylabel('Predicted IRI', fontweight='bold')
        axes[0, 0].set_title('Predictions vs Actual', fontweight='bold')
        axes[0, 0].legend()

        # Erreurs
        axes[0, 1].hist(errors, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(0, color='red', linestyle='--', lw=2)
        axes[0, 1].set_xlabel('Error (mm/m)', fontweight='bold')
        axes[0, 1].set_ylabel('Frequency', fontweight='bold')
        axes[0, 1].set_title('Error Distribution', fontweight='bold')

        # MAE par segment
        mae_per_road = np.abs(y_true - y_pred).mean(axis=0).flatten()
        axes[1, 0].bar(range(len(mae_per_road)), mae_per_road, alpha=0.7)
        axes[1, 0].axhline(mae_per_road.mean(), color='red', linestyle='--',
                           label=f'Mean: {mae_per_road.mean():.3f}')
        axes[1, 0].set_xlabel('Road Segment', fontweight='bold')
        axes[1, 0].set_ylabel('MAE (mm/m)', fontweight='bold')
        axes[1, 0].set_title('MAE per Segment', fontweight='bold')
        axes[1, 0].legend()

        # Série temporelle
        axes[1, 1].plot(y_true[:, 0, 0], 'b-', label='Actual', lw=2, marker='o')
        axes[1, 1].plot(y_pred[:, 0, 0], 'r--', label='Predicted', lw=2, marker='x')
        axes[1, 1].set_xlabel('Time Step', fontweight='bold')
        axes[1, 1].set_ylabel('IRI (mm/m)', fontweight='bold')
        axes[1, 1].set_title('Time Series (Sample)', fontweight='bold')
        axes[1, 1].legend()

        plt.tight_layout()
        plt.savefig(self.output_dir / 'evaluation_results.eps',
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Sauvegardé")

    def plot_propagation(self, model, X_test, scaler_y, A_similarity,
                         distance_matrix, source_idx=0):
        """Figure 4: Propagation spatiale."""
        print("\n🗺️  Génération spatial_propagation.eps...")

        # Baseline vs Modified
        y_baseline = model.stgt(X_test, A_similarity, training=False).numpy()
        X_modified = X_test.copy()
        X_modified[:, source_idx, :, 4] += 1.0  # IRI +1.0
        y_modified = model.stgt(X_modified, A_similarity, training=False).numpy()

        impact = scaler_y.inverse_transform(
            (y_modified - y_baseline).reshape(-1, 1)
        ).reshape(y_modified.shape).mean(axis=0).flatten()

        distances = distance_matrix[source_idx, :]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Distance vs Impact
        axes[0].scatter(distances, np.abs(impact), alpha=0.6, s=50)
        axes[0].axhline(0.1, color='red', linestyle='--',
                        label='Threshold (0.1 mm/m)')
        axes[0].set_xlabel('Distance (km)', fontweight='bold')
        axes[0].set_ylabel('|Impact| (mm/m)', fontweight='bold')
        axes[0].set_title('Spatial Propagation', fontweight='bold')
        axes[0].legend()

        # Heatmap
        impact_2d = impact.reshape(10, 10) if len(impact) == 100 else impact.reshape(-1, 1)
        im = axes[1].imshow(impact_2d, cmap='RdYlGn_r', aspect='auto')
        axes[1].set_title('Impact Heatmap', fontweight='bold')
        plt.colorbar(im, ax=axes[1], label='Impact (mm/m)')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'spatial_propagation.eps',
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Sauvegardé")


# ============================================================================
# ENTRAÎNEMENT ET ÉVALUATION
# ============================================================================

class Trainer:
    """Gestionnaire d'entraînement."""

    def __init__(self, config: Config):
        self.config = config
        config.MODEL_DIR.mkdir(exist_ok=True)

    def train(self, model, X_train, y_train, X_val, y_val):
        """Entraîne le modèle."""
        print("\n" + "=" * 80)
        print("🚀 ENTRAÎNEMENT")
        print("=" * 80)

        model.compile(
            optimizer=keras.optimizers.Adam(self.config.LEARNING_RATE),
            loss='mse',
            metrics=['mae']
        )

        callback_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=self.config.REDUCE_LR_FACTOR,
                patience=self.config.REDUCE_LR_PATIENCE,
                min_lr=self.config.MIN_LR,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                self.config.MODEL_DIR / 'best_model.weights.h5',
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=True,
                verbose=1
            ),
            callbacks.CSVLogger(self.config.MODEL_DIR / 'history.csv')
        ]

        print(f"\n⚙️  Configuration: epochs={self.config.EPOCHS}, "
              f"batch={self.config.BATCH_SIZE}, lr={self.config.LEARNING_RATE}")
        print("\n🏋️  Début entraînement...\n")

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config.EPOCHS,
            batch_size=self.config.BATCH_SIZE,
            callbacks=callback_list,
            verbose=1
        )

        print("\n✓ Entraînement terminé!")
        return history

    def evaluate(self, model, X_test, y_test, scaler_y):
        """Évalue le modèle."""
        print("\n" + "=" * 80)
        print("📊 ÉVALUATION")
        print("=" * 80)

        y_pred = model.predict(X_test, verbose=0)

        y_test_real = scaler_y.inverse_transform(y_test.reshape(-1, 1)).reshape(y_test.shape)
        y_pred_real = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).reshape(y_pred.shape)

        metrics = {
            'mae': float(mean_absolute_error(y_test_real.flatten(), y_pred_real.flatten())),
            'rmse': float(np.sqrt(mean_squared_error(y_test_real.flatten(), y_pred_real.flatten()))),
            'mape': float(mean_absolute_percentage_error(y_test_real.flatten(), y_pred_real.flatten()) * 100),
            'r2': float(r2_score(y_test_real.flatten(), y_pred_real.flatten()))
        }

        print("\n" + "=" * 60)
        print("📊 RÉSULTATS FINAUX")
        print("=" * 60)
        for k, v in metrics.items():
            unit = ' mm/m' if k in ['mae', 'rmse'] else ('%' if k == 'mape' else '')
            print(f"{k.upper():5s}: {v:.4f}{unit}")
        print("=" * 60)

        return metrics, y_pred_real, y_test_real


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Pipeline complet."""

    parser = argparse.ArgumentParser(description='STGT Ultimate (FIXED)')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Configuration
    config = Config()
    config.EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.LEARNING_RATE = args.learning_rate

    set_seed(args.seed)

    print("\n" + "=" * 80)
    print("🚀 STGT ULTIMATE - PIPELINE COMPLET (VERSION CORRIGÉE)")
    print("=" * 80)

    # 1. Chargement
    loader = DataLoader(config.DATA_DIR)
    data = loader.load_all()

    # 2. Préparation
    preprocessor = DataPreprocessor(config.WINDOW_SIZE, config.HORIZON)
    X, y, features = preprocessor.create_sequences(data['df'])
    splits = preprocessor.split_temporal(X, y, config.TRAIN_RATIO, config.VAL_RATIO)
    normalized = preprocessor.standardize(splits['train'], splits['val'], splits['test'])

    # 3. Modèle
    print("\n" + "=" * 80)
    print("🏗️  CRÉATION MODÈLE")
    print("=" * 80)

    stgt = STGTModel(config)
    model = STGTWrapper(stgt, data['A_similarity'])
    _ = model(normalized['train'][0][:1])

    print(f"\n✓ Modèle créé: {model.count_params():,} paramètres")

    # 4. Entraînement
    trainer = Trainer(config)
    history = trainer.train(model, *normalized['train'], *normalized['val'])

    # 5. Évaluation
    metrics, y_pred, y_test = trainer.evaluate(
        model, *normalized['test'], normalized['scalers'][1]
    )

    # 6. Sauvegarde
    print("\n💾 Sauvegarde résultats...")
    with open(config.MODEL_DIR / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    with open(config.MODEL_DIR / 'config.json', 'w') as f:
        json.dump(vars(config), f, indent=2, default=str)
    for i, scaler in enumerate(normalized['scalers']):
        with open(config.MODEL_DIR / f'scaler_{"X" if i == 0 else "y"}.pkl', 'wb') as f:
            pickle.dump(scaler, f)
    np.save(config.MODEL_DIR / 'predictions.npy', y_pred)
    np.save(config.MODEL_DIR / 'actuals.npy', y_test)
    print("✓ Sauvegardé")

    # 7. Visualisations
    if args.visualize:
        print("\n" + "=" * 80)
        print("🎨 VISUALISATIONS")
        print("=" * 80)

        viz = Visualizer(config.FIGURE_DIR)
        latest_iri = data['df'].groupby('road_index')['IRI'].last().values
        viz.plot_network(data['road_network'], data['coords'], latest_iri)
        viz.plot_training(pd.read_csv(config.MODEL_DIR / 'history.csv'))
        viz.plot_evaluation(y_test, y_pred)
        viz.plot_propagation(model, normalized['test'][0], normalized['scalers'][1],
                             data['A_similarity'], data['distance_matrix'])

        print("\n✓ Toutes les visualisations générées!")

    # Résumé
    print("\n" + "=" * 80)
    print("✅ PIPELINE TERMINÉ")
    print("=" * 80)
    print(f"\n📊 Performances: MAE={metrics['mae']:.4f}, R²={metrics['r2']:.4f}")
    print(f"📁 Fichiers: models/, figures/")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
