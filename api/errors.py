"""
Structured API error types + FastAPI exception handlers.

Design goal (per the integration spec): every error the client sees is
`{"error", "message", "details"}` JSON with a sane HTTP status code -
never a raw Python traceback, and never a silently-swallowed 500. Every
AnalysisAPIError subclass corresponds to one of the error codes
API_CONTRACT.md's frontend (src/api/analyzeVideo.js) already has
built-in copy for: missing_checkpoint, unsupported_media_type,
corrupt_video, feature_extraction_failed, inference_failed. A few
extra codes (missing_video, not_found, internal_error) are added for
cases the contract's built-in-copy list doesn't name but
analyzeVideo.js already handles generically (it shows message/details
verbatim for unrecognized codes).

This module does not know about the pipeline's own exception types
(MissingCheckpointError, MissingFeatureFileError, FeatureShapeError,
etc. from inference/full_pipeline.py) - server.py is responsible for
catching those and translating them into the AnalysisAPIError
subclasses defined here. Keeping that translation in server.py (the
one place that calls both the pipeline and this module) avoids this
module importing torch/pipeline code just to define error classes.
"""

import logging
import traceback

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("deepfake_api")


class AnalysisAPIError(Exception):
    """Base class for every error this API deliberately raises and
    reports to the client as structured JSON (as opposed to an
    unexpected exception, which the catch-all handler below reports as
    a generic internal_error without leaking its details)."""

    status_code = 500
    error_code = "internal_error"

    def __init__(self, message, details=None):
        super().__init__(message)
        self.message = message
        self.details = details

    def to_body(self):
        return {"error": self.error_code, "message": self.message, "details": self.details}


class MissingVideoError(AnalysisAPIError):
    status_code = 400
    error_code = "missing_video"


class UnsupportedMediaTypeError(AnalysisAPIError):
    status_code = 415
    error_code = "unsupported_media_type"


class FileTooLargeError(AnalysisAPIError):
    status_code = 413
    error_code = "file_too_large"


class CorruptVideoError(AnalysisAPIError):
    status_code = 422
    error_code = "corrupt_video"


class FeatureExtractionFailedError(AnalysisAPIError):
    status_code = 500
    error_code = "feature_extraction_failed"


class InferenceFailedError(AnalysisAPIError):
    status_code = 500
    error_code = "inference_failed"


class MissingCheckpointAPIError(AnalysisAPIError):
    """Maps inference/full_pipeline.py's MissingCheckpointError to a 503:
    this is not a bug and not the client's fault - a required trained
    checkpoint genuinely does not exist on this server yet."""

    status_code = 503
    error_code = "missing_checkpoint"


class NotFoundError(AnalysisAPIError):
    status_code = 404
    error_code = "not_found"


def install_error_handlers(app: FastAPI):
    @app.exception_handler(AnalysisAPIError)
    async def handle_analysis_error(request: Request, exc: AnalysisAPIError):
        if exc.status_code >= 500:
            logger.error("%s on %s: %s (%s)", exc.error_code, request.url.path, exc.message, exc.details)
        else:
            logger.info("%s on %s: %s", exc.error_code, request.url.path, exc.message)
        return JSONResponse(status_code=exc.status_code, content=exc.to_body())

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception):
        # Full traceback goes to the server log only - never to the
        # client. The client gets a generic, safe message plus an
        # error code the frontend already treats as "show message
        # verbatim" for unrecognized codes.
        logger.error("Unhandled exception on %s:\n%s", request.url.path, traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_error",
                "message": "Something went wrong processing this request.",
                "details": None,
            },
        )
