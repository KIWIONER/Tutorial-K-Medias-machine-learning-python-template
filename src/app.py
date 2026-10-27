import os
import urllib.request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

def main():
    print("=== Pipeline de Machine Learning: California Housing Clustering & Classification ===")

    # 1. Definir rutas
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(base_dir, ".."))
    
    raw_dir = os.path.join(project_dir, "data", "raw")
    processed_dir = os.path.join(project_dir, "data", "processed")
    models_dir = os.path.join(project_dir, "models")
    
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    raw_data_path = os.path.join(raw_dir, "housing.csv")
    data_url = "https://raw.githubusercontent.com/4GeeksAcademy/k-means-project-tutorial/main/housing.csv"

    # 2. Descargar dataset si no existe
    if not os.path.exists(raw_data_path):
        print(f"Descargando dataset desde {data_url}...")
        urllib.request.urlretrieve(data_url, raw_data_path)
        print("Descarga completada.")
    else:
        print(f"Dataset encontrado en: {raw_data_path}")

    # 3. Carga y selección de columnas de interés
    df_raw = pd.read_csv(raw_data_path)
    features = ["Latitude", "Longitude", "MedInc"]
    df = df_raw[features].copy()
    print(f"Dataset cargado con {len(df)} registros y características: {features}")

    # 4. División Train / Test
    X_train, X_test = train_test_split(df, test_size=0.2, random_state=42)
    print(f"Train set: {X_train.shape} | Test set: {X_test.shape}")

    # 5. Paso 2: Modelo K-Means (No supervisado)
    print("\n--- Entrenando K-Means (k=6) ---")
    kmeans = KMeans(n_clusters=6, random_state=42, n_init="auto")
    train_clusters = kmeans.fit_predict(X_train)
    
    X_train_clustered = X_train.copy()
    X_train_clustered["cluster"] = train_clusters
    X_train_clustered["cluster"] = X_train_clustered["cluster"].astype("category")

    # 6. Paso 3: Predicción en Test con K-Means
    test_clusters = kmeans.predict(X_test)
    X_test_clustered = X_test.copy()
    X_test_clustered["cluster"] = test_clusters
    X_test_clustered["cluster"] = X_test_clustered["cluster"].astype("category")

    # 7. Paso 4: Modelo Supervisado (Random Forest)
    print("\n--- Entrenando Clasificador Supervisado (Random Forest) ---")
    X_tr = X_train[features]
    y_tr = X_train_clustered["cluster"]
    X_te = X_test[features]
    y_te = X_test_clustered["cluster"]

    clf_rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    clf_rf.fit(X_tr, y_tr)

    acc_train = accuracy_score(y_tr, clf_rf.predict(X_tr))
    y_pred_test = clf_rf.predict(X_te)
    acc_test = accuracy_score(y_te, y_pred_test)

    print(f"Accuracy en Train: {acc_train:.4f} ({acc_train*100:.2f}%)")
    print(f"Accuracy en Test:  {acc_test:.4f} ({acc_test*100:.2f}%)")
    print("\nReporte de clasificación (Test):")
    print(classification_report(y_te, y_pred_test))

    # 8. Paso 5: Guardar modelos y datos procesados
    print("\n--- Guardando artefactos ---")
    kmeans_file = os.path.join(models_dir, "kmeans_clusters_6.joblib")
    rf_file = os.path.join(models_dir, "random_forest_classifier.joblib")

    joblib.dump(kmeans, kmeans_file)
    joblib.dump(clf_rf, rf_file)
    print(f"Modelo K-Means guardado en: {kmeans_file}")
    print(f"Modelo Random Forest guardado en: {rf_file}")

    train_proc_file = os.path.join(processed_dir, "housing_train_clustered.csv")
    test_proc_file = os.path.join(processed_dir, "housing_test_clustered.csv")
    X_train_clustered.to_csv(train_proc_file, index=False)
    X_test_clustered.to_csv(test_proc_file, index=False)
    print(f"Datos procesados guardados en: {processed_dir}")

    print("\n=== ¡Pipeline finalizado con éxito! ===")

if __name__ == "__main__":
    main()
