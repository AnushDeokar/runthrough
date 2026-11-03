"""The ``simulated_conversation`` solver."""

from __future__ import annotations

from typing import Literal
from uuid import uuid4

from inspect_ai.model import ChatMessageUser, Model, ModelOutput, get_model
from inspect_ai.solver import Generate, Solver, TaskState, solver

from runthrough.adapters.base import Target, TurnContext
from runthrough.scenario import scenario_from_metadata
from runthrough.simulator.convert import (
    RETRY_INSTRUCTION,
    clean_user_message,
    looks_like_framing,
    simulator_view,
    strip_framing,
    turn_to_messages,
)
from runthrough.simulator.prompts import DEFAULT_SIMULATOR_PROMPT, STOP_TOKEN

RESULT_KEY = "runthrough"
"""Key in ``TaskState.metadata`` where the solver records how the conversation went."""

StopReason = Literal["simulator_stopped", "max_turns", "agent_error"]


@solver
def simulated_conversation(
    target: Target,
    *,
    simulator: str | Model | None = None,
    max_turns: int | None = None,
    prompt: str = DEFAULT_SIMULATOR_PROMPT,
) -> Solver:
    """Hold a conversation between a simulated user and the agent under test.

    Each turn, the simulated user writes a message (scripted ``opening`` messages
    first, then the simulator model), the target sends it to the agent, and the
    agent's reply, tool calls included, is appended to ``state.messages``.

    The conversation ends when the simulator replies with the stop token, the
    agent returns an error, or the turn limit is reached.

    Args:
        target: The agent endpoint to talk to.
        simulator: Model that plays the user. Defaults to the ``simulator`` model
            role, then to the eval's main model.
        max_turns: Override every scenario's ``max_turns``.
        prompt: System prompt for the simulator. Must contain ``{persona}``,
            ``{goal}`` and ``{stop_token}``.
    """
    for field in ("{persona}", "{goal}", "{stop_token}"):
        if field not in prompt:
            raise ValueError(f"Simulator prompt must contain {field}")

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        scenario = scenario_from_metadata(state.metadata)
        model = get_model(simulator, role="simulator")
        system_prompt = prompt.format(
            persona=scenario.persona, goal=scenario.goal, stop_token=STOP_TOKEN
        )
        limit = max_turns or scenario.max_turns
        conversation_id = str(uuid4())

        # The sample input only describes the scenario; the transcript starts empty.
        state.messages = []
        stop_reason: StopReason = "max_turns"
        turns = 0

        for turn in range(1, limit + 1):
            if turn <= len(scenario.opening):
                user_text = scenario.opening[turn - 1]
            else:
                user_text = await _next_user_message(model, system_prompt, state.messages)
                if STOP_TOKEN in user_text or not user_text:
                    stop_reason = "simulator_stopped"
                    break

            state.messages.append(ChatMessageUser(content=user_text))
            history = state.messages if target.history == "full" else state.messages[-1:]
            reply = await target.send(
                list(history),
                TurnContext(conversation_id=conversation_id, turn=turn, context=scenario.context),
            )
            state.messages.extend(turn_to_messages(reply))
            turns = turn
            if reply.error is not None:
                stop_reason = "agent_error"
                break

        state.metadata[RESULT_KEY] = {
            "conversation_id": conversation_id,
            "turns": turns,
            "stop_reason": stop_reason,
        }
        state.output = ModelOutput.from_content(
            model="runthrough/target", content=_last_reply(state)
        )
        return state

    return solve


async def _next_user_message(model: Model, system_prompt: str, conversation: list) -> str:
    """Ask the simulator for the next message, retrying once if it echoes the transcript."""
    view = simulator_view(system_prompt, conversation)
    output = await model.generate(view)
    text = clean_user_message(output.completion)
    if not looks_like_framing(text):
        return text
    retry = [*view, output.message, ChatMessageUser(content=RETRY_INSTRUCTION)]
    text = clean_user_message((await model.generate(retry)).completion)
    return strip_framing(text)


def _last_reply(state: TaskState) -> str:
    for message in reversed(state.messages):
        if message.role == "assistant" and message.text:
            return message.text
    return ""
