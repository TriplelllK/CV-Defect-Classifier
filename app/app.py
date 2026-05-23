import os
from flask import Flask, render_template, request, redirect, flash, jsonify

from .model_utils import predict_defect, model

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "dev-secret")
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


def _get_file():
    if "file" not in request.files:
        return None, "Нет поля 'file' в запросе"
    file = request.files["file"]
    if file.filename == "":
        return None, "Файл не выбран"
    return file, None


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file, err = _get_file()
        if err:
            flash(err)
            return redirect(request.url)
        try:
            pred_class, confidence, probs = predict_defect(file)
            return render_template("index.html", pred_class=pred_class,
                                   confidence=confidence, probs=probs)
        except Exception as e:
            flash(f"Не удалось обработать файл: {e}")
            return redirect(request.url)
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    file, err = _get_file()
    if err:
        return jsonify({"success": False, "error": err}), 400
    try:
        pred_class, confidence, probs = predict_defect(file)
        return jsonify({
            "success": True,
            "prediction": {
                "class": pred_class,
                "confidence": confidence,
                "probabilities": probs,
            },
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=os.environ.get("FLASK_DEBUG") == "1")
