"""Interactive text console with safe, explicit development feedback."""

import json
from uuid import uuid4

from .agent import AgentOutputError
from .llm import LLMError
from .schemas import CustomerInput

HELP = ("Try: 'track my order', 'explain the refund policy', 'return an item', "
        "'place a new order', or 'human agent'.\n"
        "Commands: /status shows mode and the last result; /reset starts over; /help shows examples; /quit exits.")


def run_chat(agent, *, debug=False):
    session = uuid4().hex
    last_result = None
    mode = "live_policy" if agent.rag else "offline"
    print("Stateful support chat. " + ("Live RAG is enabled for policy questions." if agent.rag else
          "Offline mode: local rules and tools are available; Gemini is disabled."))
    print(HELP)
    try:
        while True:
            try:
                message = input("> ").strip()
            except EOFError:
                break
            if message == "/quit":
                break
            if message == "/help":
                print(HELP)
                continue
            if message == "/status":
                print(json.dumps({"mode": mode, "local_support": "available", "llm_policy_generation": "enabled" if agent.rag else "disabled",
                    "order_placement": "unavailable", "last_result": last_result}, ensure_ascii=False, indent=2))
                continue
            if message == "/reset":
                agent.clear_session(session)
                session, last_result = uuid4().hex, None
                print("Started a fresh conversation; order verification has been cleared.")
                continue
            if message.startswith("/"):
                print("Unknown command. Use /help, /status, /reset, or /quit.")
                continue
            if not message:
                print("Please type a support question, or use /help for examples.")
                continue
            try:
                response = agent.respond(CustomerInput(message, session))
                last_result = response.diagnostics
                print(response.reply)
            except (LLMError, AgentOutputError) as exc:
                # Never echo raw provider bodies, model text or proof-bearing input.
                last_result = {"status": "failed", "code": exc.code, "mode": mode,
                               "stage": "policy_generation", "llm_used": True}
                print("Policy generation failed; this turn has no valid generated answer. You can retry or use /status for the error code.")
            if debug:
                print("[debug] " + json.dumps(last_result, ensure_ascii=False))
    finally:
        agent.clear_session(session)
