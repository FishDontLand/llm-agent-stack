"""Chess tool implementations, decoupled from the agent that registers them.

Every function here takes the HTTP client explicitly instead of reading it off
an agent, so the same code can run in the agent process or inside the sandbox
beside the server it talks to.
"""

from __future__ import annotations

import base64
import json
from typing import Any

import httpx

CHESS_PORT = 8000


def _request_state(
    client: httpx.Client, method: str, endpoint: str, **kwargs: Any
) -> dict[str, Any]:
    """Make one chess API request and validate its JSON response."""

    response = client.request(method, endpoint, **kwargs)
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"Chess server returned non-JSON ({response.status_code})."
        ) from exc
    if response.status_code >= 400:
        detail = (
            payload.get("detail", payload) if isinstance(payload, dict) else payload
        )
        raise ValueError(str(detail))
    if not isinstance(payload, dict):
        raise RuntimeError("Chess server response must be a JSON object.")
    return payload


def _simulate_move(client: httpx.Client, arguments: str) -> str:
    """New tool: inspect FEN or simulate one ply without changing the game.

    Takes the raw JSON arguments of one tool call and returns the observation
    to send back, so a bad argument or a server error reaches the model as a
    recoverable ``<chess_error>`` instead of ending the run.
    """
    content = None
    try:
        arguments = json.loads(arguments)
    except Exception as e:
        content = f"<chess_error>Can't convert argument to json object, error: {e}</chess_error><arguments>{arguments}</arguments>"

    if content is not None:
        return f"<chess_error>{content}</chess_error>"

    fen = arguments.get("fen")
    if fen is None:
        return "<chess_error>Missing fen</chess_error>"
    
    move = arguments.get("move")

    if move is None:
        res = client.post('/api/simulate', json={
            'fen': fen
        })
    else:
        res = client.post('/api/simulate', json={
            'fen': fen,
            'move': move
        })

    if res.is_success:
        return res.text

    error = f"<chess_error>{res.text}</chess_error><argument>{arguments}</argument>"
    return error


def _play_move(client: httpx.Client, arguments: str) -> str:
    """Existing tool: play one move as White and return the resulting state.

    Takes the raw JSON arguments of one tool call. Returns the new state, or a
    `<chess_error>` observation if the move could not be played.
    """
    content = None
    try:
        arguments = json.loads(arguments)
    except Exception as e:
        content = f"<chess_error>Can't convert argument to json object, error: {e}</chess_error><argument>{arguments}</argument>"

    if content is not None:
        return f"<chess_error>{content}</chess_error>"

    move = arguments.get("move")
    if move is None:
        return f"<chess_error>Missing move argument</chess_error>"

    if type(move) is not str:
        return f"<chess_error>expecting mvoe argument to be str, but receive {type(move)}</chess_error>"

    res = client.post('/api/move', json={"move": move})
    
    if res.is_success:
        return res.text

    return f"<chess_error>{res.text}</chess_error>"




def _run_python(env: Any, port: int, arguments: str) -> str:
    """New tool: run Python with access to the existing registered tools.

    The snippet runs inside the sandbox, which already has the tool
    implementations and the chess server, so code the model wrote never
    executes in the agent process.
    """
    content = None
    try:
        arguments = json.loads(arguments)
    except Exception as e:
        return f"<chess_error>Can't parse arguments, got error: {e}</chess_error><argument>{arguments}</argument>"

    code = arguments.get("code")
    if code is None:
        return "<chess_error>No python code for execution</chess_error>"

    if type(code) is not str:
        return f"<chess_error>Expecting the code snippet in the format of a string, but found {type(code)}</chess_error>"

    encoded_code = base64.b64encode(code.encode()).decode('utf-8')
    exec_result = env.execute(
        command=f"python /opt/agent_harness/sandbox_python.py {port} {encoded_code}",
    )

    if exec_result.get("returncode", -1) != 0:
        print("code: ", code)
        print("sandbox output: ", exec_result)
        return f"<chess_error>{exec_result["stderr"]}</chess_error>"

    output = exec_result.get("stdout", "")
    output_json = json.loads(output)
            
    if output_json.get("stderr", ""):
        error = output_json['stderr']
        return f"<chess_error>{error}</chess_error>"

    if output_json.get("stdout", ""):
        return f"<stdout>{output_json['stdout']}</stdout>"

    print("can't reach this step: ", exec_result)
    raise Exception("Impossible to happen")
    


def _invoke_skill(skills: dict[str, dict[str, str]], arguments: str) -> str:
    """Existing tool: load one skill's instructions into the conversation."""
    try:
        arguments = json.loads(arguments)
    except Exception as e:
         return f"<chess_error>Can't parse arguments, got error: {e}</chess_error><argument>{arguments}</argument>"

    skill_name = arguments.get("name")

    if skill_name is None:
         return f"<chess_error>skill name is missing</chess_error>"

    if type(skill_name) is not str:
         return f"<chess_error>Expecting skill name to be a string, but found {type(skill_name)}</chess_error>"

    skill_details = skills.get(skill_name, {}).get('content', '')
    return f"<skill>\n{skill_details}\n</skill>"

def _game_state(client: httpx.Client, reset: bool = False) -> dict:
    """Read the live game, or start a new one and read the opening position."""

    method, endpoint = ("POST", "/api/reset") if reset else ("GET", "/api/state")
    return _request_state(client, method, endpoint)
