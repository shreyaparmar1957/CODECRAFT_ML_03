import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import joblib

# If your dataset folder is beside this script, this works as-is.
DATASET_DIR = r"C:\Users\shrey\PycharmProjects\pythonProject\CODECRAFT\ml_03\dataset\train"
import os
print("Working dir:", os.getcwd())
print("Contents:", os.listdir(os.path.dirname(__file__)))
print("Dataset path exists:", os.path.exists(DATASET_DIR))
if os.path.exists(DATASET_DIR):
    print("Train subfolders:", os.listdir(DATASET_DIR))


def load_images(base_dir):
    X, y = [], []
    labels = {"cats": 0, "dogs": 1}
    for label, value in labels.items():
        folder = os.path.join(base_dir, label)
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Missing folder: {folder}")
        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)
            try:
                img = Image.open(img_path).convert("L").resize((64, 64))
                X.append(np.array(img).flatten())
                y.append(value)
            except Exception:
                # skip unreadable/corrupt files
                pass
    return np.array(X), np.array(y)

def main():
    print("Loading images from:", DATASET_DIR)
    X, y = load_images(DATASET_DIR)
    print(f"Loaded {len(X)} images. Cats+Dogs.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    print("Training SVM...")
    model = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.3f}")
    print("\nClassification report:\n", classification_report(y_test, y_pred, target_names=["Cat","Dog"]))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    # Save model
    joblib.dump({"model": model, "scaler": scaler}, "svm_cats_dogs.joblib")
    print("\nSaved model → svm_cats_dogs.joblib")

    # Quick preview: a few random predictions
    try:
        import random
        idxs = np.random.choice(len(X_test), size=min(5, len(X_test)), replace=False)
        fig, axes = plt.subplots(1, len(idxs), figsize=(12, 4))
        for ax, i in zip(axes, idxs):
            ax.imshow(X_test[i].reshape(64, 64), cmap="gray")
            ax.set_title(f"Pred: {'Cat' if y_pred[i]==0 else 'Dog'}")
            ax.axis("off")
        plt.tight_layout()
        plt.show()
    except Exception:
        pass

if __name__ == "__main__":
    main()
