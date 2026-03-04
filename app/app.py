from flask import (
    Flask,
    render_template,
    request,
    redirect,
    flash,
    jsonify,
)
from .model_utils import predict_defect

app = Flask(__name__)
app.secret_key = "super-secret-key"  # TODO: поменять в проде


@app.route("/", methods=["GET", "POST"])
def index():
    # Страница с загрузкой файла
    if request.method == "POST":
        if "file" not in request.files:
            flash("Файл не найден в запросе.")
            return redirect(request.url)

        file = request.files["file"]

        if file.filename == "":
            flash("Файл не выбран.")
            return redirect(request.url)

        try:
            pred_class, confidence, probs_dict = predict_defect(file)
            return render_template(
                "index.html",
                pred_class=pred_class,
                confidence=confidence,
                probs=probs_dict,
            )
        except Exception as e:
            flash(f"Ошибка при обработке файла: {e}")
            return redirect(request.url)

    # GET-запрос
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
        # API для предсказания по изображению
    if "file" not in request.files:
        return jsonify(
            {
                "success": False,
                "error": "Поле 'file' не найдено в запросе",
            }
        ), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify(
            {
                "success": False,
                "error": "Файл не выбран",
            }
        ), 400

    try:
        pred_class, confidence, probs_dict = predict_defect(file)
        return jsonify(
            {
                "success": True,
                "prediction": {
                    "class": pred_class,
                    "confidence": confidence,
                    "probabilities": probs_dict,
                },
            }
        )
    except Exception as e:
        return jsonify(
            {
                "success": False,
                "error": f"Ошибка при обработке файла: {str(e)}",
            }
        ), 500


@app.route("/api/health", methods=["GET"])
def health():
    # Проверка, что сервис запущен
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    # Локальный запуск
    app.run(host="0.0.0.0", port=5000, debug=True)
