import pickle
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'rf_severity.pkl')

def predict_severity(features):
    """
    Given extracted features, predict severity and injury info using a trained classifier.
    Returns severity label and injury label.
    Features: list/array shaped as expected by your model.
    """
    # Defensive code -- handle missing model gracefully
    try:
        with open(MODEL_PATH, 'rb') as f:
            clf = pickle.load(f)
        prediction = clf.predict([features])
        # prediction shape: [(severity_index, injury_index)], update below for your case
        severity_map = {0: "MINOR", 1: "MAJOR", 2: "CRITICAL"}
        injury_map = {0: "none", 1: "minor injury", 2: "major injury"}
        if hasattr(prediction[0], '__iter__'):
            s_idx, i_idx = prediction[0][0], prediction[0][1]
        else:
            # fallback for single-output case
            s_idx, i_idx = prediction[0], 0
        severity = severity_map.get(s_idx, "UNKNOWN")
        injury = injury_map.get(i_idx, "UNKNOWN")
        return severity, injury
    except Exception as e:
        # If error in prediction, you may log or handle fallback
        print("Severity prediction issue:", str(e))
        return "UNKNOWN", "UNKNOWN"
