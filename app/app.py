import os
import logging
from flask import Flask, render_template, request, redirect, flash, jsonify

from .model_utils import predict_defect, model

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "dev-secret")
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

logging.basicConfig(level=logging.INFO)


def _get_file():
    if "file" not in request.files:
        return None, "Нет поля 'file' в запросе"
    file = request.files["file"]
    if file.filename == "":
        return None, "Файл не выбран"
    if not (file.mimetype or "").startswith("image/"):
        return None, "Ожидается изображение"
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
        except Exception:
            app.logger.exception("predict failed")
            flash("Не удалось обработать файл")
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
    except Exception:
        app.logger.exception("api predict failed")
        return jsonify({"success": False, "error": "Внутренняя ошибка обработки"}), 500


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=os.environ.get("FLASK_DEBUG") == "1")
