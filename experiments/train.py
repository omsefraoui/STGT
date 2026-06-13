import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import os
import sys


HORIZONS = [1, 3, 6, 12]
WINDOW_LENGTH = 12
DATA_DIR = 'data/multihorizon'
RAW_DATA_FILE = 'data/stgt_data_FINAL.csv'  # Or your real data


os.makedirs(DATA_DIR, exist_ok=True)


def load_data(filepath):
    
    print(f"\n{'=' * 70}")
    print(f"Chargement données depuis: {filepath}")
    print(f"{'=' * 70}")

    df = pd.read_csv(filepath)

   
    segments = sorted(df['segment_id'].unique())
    months = sorted(df['month'].unique())

    N = len(segments)
    T_total = len(months)

    print(f" Loaded {N} segments × {T_total} months")

    # Feature columns (everything except segment_id, month, IRI)
    feature_cols = [col for col in df.columns
                    if col not in ['segment_id', 'month', 'IRI']]
    F = len(feature_cols)

    print(f"✓ Feature dimensions: F = {F}")
    print(f"✓ Feature columns: {feature_cols[:5]}... (showing first 5)")

   
    X = np.zeros((N, T_total, F))
    y = np.zeros((N, T_total))

    for i, seg_id in enumerate(segments):
        seg_data = df[df['segment_id'] == seg_id].sort_values('month')
        X[i] = seg_data[feature_cols].values
        y[i] = seg_data['IRI'].values

    print(f" Data shapes: X {X.shape}, y {y.shape}")

    return X, y, N, T_total, F


def prepare_data_multihorizon(X, y, T=12, horizons=[1, 3, 6, 12]):
   

    N, T_total, F = X.shape
    dict_horizons = {}

    for H in horizons:
        print(f"\n{'=' * 70}")
        print(f"Préparation pour H = {H} mois")
        print(f"{'=' * 70}")

        X_h = []
        y_h = []

      
        for i in range(N):
            for t in range(T_total - T - H + 1):
                window = X[i, t:t + T, :]  # (T, F)
                target = y[i, t + T + H - 1]  # IRI à t+T+H

                X_h.append(window)
                y_h.append(target)

        X_h = np.array(X_h)  # (num_samples, T, F)
        y_h = np.array(y_h)  # (num_samples,)

        print(f"Dataset shape: X {X_h.shape}, y {y_h.shape}")
        print(f"Samples per segment: {len(X_h) // N}")

        
        num_samples = len(X_h)
        train_split = int(0.70 * num_samples)
        val_split = int(0.85 * num_samples)

        X_train = X_h[:train_split]
        y_train = y_h[:train_split]

        X_val = X_h[train_split:val_split]
        y_val = y_h[train_split:val_split]

        X_test = X_h[val_split:]
        y_test = y_h[val_split:]

        print(f"Split: Train {X_train.shape[0]}, Val {X_val.shape[0]}, Test {X_test.shape[0]}")

       
        scaler = StandardScaler()
        scaler.fit(X_train.reshape(-1, F))

        X_train_scaled = scaler.transform(X_train.reshape(-1, F)).reshape(X_train.shape)
        X_val_scaled = scaler.transform(X_val.reshape(-1, F)).reshape(X_val.shape)
        X_test_scaled = scaler.transform(X_test.reshape(-1, F)).reshape(X_test.shape)

        
        print(f"Target y statistics (H={H}):")
        print(f"  Mean: {y_h.mean():.3f}, Std: {y_h.std():.3f}")
        print(f"  Min: {y_h.min():.3f}, Max: {y_h.max():.3f}")

        dict_horizons[H] = {
            'X_train': X_train_scaled,
            'y_train': y_train,
            'X_val': X_val_scaled,
            'y_val': y_val,
            'X_test': X_test_scaled,
            'y_test': y_test,
            'scaler': scaler,
            'N': X_train.shape[0],  # num segments in this batch
            'F': F  # num features
        }

    return dict_horizons


def save_multihorizon_data(dict_horizons, output_dir='data/multihorizon/'):
    
    os.makedirs(output_dir, exist_ok=True)

    for H, data in dict_horizons.items():
        filepath = os.path.join(output_dir, f'data_H{H}.pkl')
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"✓ Saved H={H} data to {filepath}")


if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("PHASE 1: DATA PREPARATION FOR MULTI-HORIZON")
    print("=" * 70)

    try:
        # Load original data
        X, y, N, T_total, F = load_data(RAW_DATA_FILE)

        # Prepare for all horizons
        dict_horizons = prepare_data_multihorizon(X, y, T=WINDOW_LENGTH, horizons=HORIZONS)

        # Save to disk
        save_multihorizon_data(dict_horizons)

        print("\n" + "=" * 70)
        print(" PHASE 1 COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"Ready for Phase 2: Training")

    except Exception as e:
        print(f"\n ERROR: {e}")
        import traceback

        traceback.print_exc()
