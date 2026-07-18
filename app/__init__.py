import logging
import sys

from flask import Flask, jsonify
from flask_cors import CORS

from app.config import CORS_ORIGINS, LOG_LEVEL, STATIC_DIR, TEMPLATE_DIR
from app.model import initialize_model
from app.routes import bp


def configure_logging() -> None:
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def create_app() -> Flask:
    configure_logging()
    logger = logging.getLogger(__name__)

    application = Flask(
        __name__,
        template_folder=str(TEMPLATE_DIR),
        static_folder=str(STATIC_DIR),
    )

    if CORS_ORIGINS.strip() == "*":
        CORS(application)
    else:
        origins = [origin.strip() for origin in CORS_ORIGINS.split(",") if origin.strip()]
        CORS(application, resources={r"/*": {"origins": origins}})

    application.register_blueprint(bp)

    @application.errorhandler(404)
    def not_found(error):
        if request_wants_json():
            return jsonify({"error": "Not found"}), 404
        return "Not found", 404

    @application.errorhandler(500)
    def internal_error(error):
        logger.exception("Unhandled server error.")
        if request_wants_json():
            return jsonify({"error": "Internal server error"}), 500
        return "Internal server error", 500

    @application.errorhandler(Exception)
    def handle_exception(error):
        from werkzeug.exceptions import HTTPException

        if isinstance(error, HTTPException):
            return error

        logger.exception("Unhandled exception: %s", error)
        if request_wants_json():
            return jsonify({"error": "Internal server error"}), 500
        return "Internal server error", 500

    try:
        initialize_model()
    except Exception:
        logger.exception("Failed to initialize model during startup.")
        raise

    logger.info("Application startup complete.")
    return application


def request_wants_json() -> bool:
    from flask import request

    return (
        request.path.startswith("/predict")
        or request.path.startswith("/health")
        or request.accept_mimetypes.best == "application/json"
    )
