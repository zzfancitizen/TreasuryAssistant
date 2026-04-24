from app.core.a2a_client import (
    build_message_send_request,
    build_message_stream_request,
    parse_message_send_response,
    parse_sse_data_line,
)


def test_build_message_send_request_uses_a2a_jsonrpc_method() -> None:
    payload = build_message_send_request("查余额", context={"trace_id": "trace-1"})

    assert payload["jsonrpc"] == "2.0"
    assert payload["method"] == "message/send"
    assert payload["params"]["metadata"]["context"] == {"trace_id": "trace-1"}
    assert payload["params"]["message"]["parts"][0]["text"] == "查余额"


def test_build_message_stream_request_uses_a2a_stream_method() -> None:
    payload = build_message_stream_request("查余额")

    assert payload["jsonrpc"] == "2.0"
    assert payload["method"] == "message/stream"


def test_parse_message_send_response_extracts_json_artifact() -> None:
    payload = {
        "jsonrpc": "2.0",
        "id": "request-1",
        "result": {
            "artifacts": [
                {
                    "parts": [
                        {
                            "kind": "text",
                            "text": '{"agent": "CashAgent", "status": "completed"}',
                        }
                    ]
                }
            ]
        },
    }

    assert parse_message_send_response(payload) == {
        "agent": "CashAgent",
        "status": "completed",
    }


def test_parse_sse_data_line_extracts_jsonrpc_event() -> None:
    event = parse_sse_data_line('data: {"jsonrpc": "2.0", "result": {"kind": "status-update"}}')

    assert event == {"jsonrpc": "2.0", "result": {"kind": "status-update"}}
