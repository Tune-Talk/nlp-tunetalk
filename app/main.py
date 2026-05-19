import logging
from datetime import datetime, timezone

from flask import Flask, request, jsonify
from flask_cors import CORS
from flasgger import Swagger

from app.services.chat_service import ChatService
from app.services.playlist_service import PlaylistService
from app.config import CANONICAL_LABELS

logger = logging.getLogger(__name__)


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)
    Swagger(app)

    chat_service = ChatService()
    playlist_service = PlaylistService()

    @app.route("/api/chat", methods=["POST"])
    def chat():
        """Chat endpoint that analyzes emotion and returns a playlist.
        ---
        tags:
          - Chat
        parameters:
          - in: body
            name: body
            required: true
            schema:
              type: object
              required:
                - user_id
                - message
              properties:
                user_id:
                  type: string
                  description: Unique identifier for the user.
                message:
                  type: string
                  description: The user's chat message (minimum 5 words required).
        responses:
          200:
            description: Successful chat response with emotion analysis and playlist.
            schema:
              type: object
              properties:
                user_id:
                  type: string
                timestamp:
                  type: string
                  format: date-time
                emotion:
                  type: object
                  properties:
                    label:
                      type: string
                    confidence:
                      type: number
                    secondary_emotion:
                      type: string
                support_response:
                  type: object
                  properties:
                    text:
                      type: string
                playlist:
                  type: object
                  properties:
                    mood_category:
                      type: string
                    songs:
                      type: array
                      items:
                        type: object
                        properties:
                          song_id:
                            type: string
                          title:
                            type: string
                          artist:
                            type: string
                          genre:
                            type: string
                          mood_tag:
                            type: string
                          spotify_url:
                            type: string
                          cover_image:
                            type: string
                    total_songs:
                      type: integer
          400:
            description: Bad request (invalid JSON or missing fields).
          422:
            description: Unprocessable entity (validation error).
          500:
            description: Internal server error.
        """
        try:
            body = request.get_json(silent=True)
            if not body:
                return _error(400, "Request body must be valid JSON.")

            user_id = body.get("user_id", "").strip()
            message = body.get("message", "").strip()

            if not user_id:
                return _error(400, "Missing required field: user_id.")
            if not message:
                return _error(400, "Missing required field: message.")

            result = chat_service.process(user_id, message)
            return jsonify(result), 200

        except ValueError as e:
            return _error(422, str(e))
        except Exception as e:
            logger.exception("Unexpected error in /api/chat")
            return _error(500, "Internal server error. Please try again later.")

    @app.route("/api/health", methods=["GET"])
    def health():
        """Health check endpoint.
        ---
        tags:
          - System
        responses:
          200:
            description: Service is healthy.
            schema:
              type: object
              properties:
                status:
                  type: string
                timestamp:
                  type: string
                  format: date-time
        """
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }), 200

    @app.route("/api/playlist", methods=["POST"])
    def playlist():
        """Get a playlist for a given emotion label.
        ---
        tags:
          - Playlist
        parameters:
          - in: body
            name: body
            required: true
            schema:
              type: object
              required:
                - emotion_label
              properties:
                emotion_label:
                  type: string
                  enum: ["sadness", "anger", "joy", "anxiety", "calm"]
                  description: The emotion to generate a playlist for.
                n:
                  type: integer
                  default: 5
                  description: Number of songs to return.
        responses:
          200:
            description: Playlist generated successfully.
            schema:
              type: object
              properties:
                mood_category:
                  type: string
                songs:
                  type: array
                  items:
                    type: object
                    properties:
                      song_id:
                        type: string
                      title:
                        type: string
                      artist:
                        type: string
                      genre:
                        type: string
                      mood_tag:
                        type: string
                      spotify_url:
                        type: string
                      cover_image:
                        type: string
                total_songs:
                  type: integer
          400:
            description: Bad request (invalid JSON or missing fields).
          422:
            description: Unprocessable entity (invalid emotion label).
          500:
            description: Internal server error.
        """
        try:
            body = request.get_json(silent=True)
            if not body:
                return _error(400, "Request body must be valid JSON.")

            emotion_label = body.get("emotion_label", "").strip()
            n = body.get("n", 5)

            if not emotion_label:
                return _error(400, "Missing required field: emotion_label.")

            if emotion_label not in CANONICAL_LABELS:
                return _error(
                    422,
                    f"Invalid emotion_label: '{emotion_label}'. "
                    f"Must be one of: {', '.join(CANONICAL_LABELS)}."
                )

            try:
                n = int(n)
            except (TypeError, ValueError):
                return _error(400, "n must be a positive integer.")

            if n < 1:
                return _error(400, "n must be a positive integer.")

            result = playlist_service.get_playlist(emotion_label, n)
            return jsonify(result), 200

        except ValueError as e:
            return _error(422, str(e))
        except Exception as e:
            logger.exception("Unexpected error in /api/playlist")
            return _error(500, "Internal server error. Please try again later.")

    @app.errorhandler(400)
    def bad_request(_e):
        return _error(400, "Bad request.")

    @app.errorhandler(404)
    def not_found(_e):
        return _error(404, "Endpoint not found.")

    @app.errorhandler(405)
    def method_not_allowed(_e):
        return _error(405, "Method not allowed.")

    @app.errorhandler(422)
    def unprocessable(_e):
        return _error(422, "Unprocessable entity.")

    @app.errorhandler(500)
    def server_error(_e):
        return _error(500, "Internal server error.")

    return app


def _error(code: int, message: str) -> tuple:
    return (
        jsonify({
            "status": "error",
            "code": code,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }),
        code,
    )
