# Copyright (c) 2026 Open Beni
# SPDX-License-Identifier: MIT

"""Filter log messages from empty WebSocket probes."""

import logging

from websockets.exceptions import InvalidMessage


def _keep_websocket_log(record: logging.LogRecord) -> bool:
    if record.getMessage() != "opening handshake failed" or not record.exc_info:
        return True
    error = record.exc_info[1]
    if not isinstance(error, InvalidMessage):
        return True
    request_error = error.__cause__
    if not isinstance(request_error, EOFError):
        return True
    stream_error = request_error.__cause__
    return not (
        str(request_error) == "connection closed while reading HTTP request line"
        and isinstance(stream_error, EOFError)
        and str(stream_error) == "stream ends after 0 bytes, before end of line"
    )


logging.getLogger("websockets.server").addFilter(_keep_websocket_log)
