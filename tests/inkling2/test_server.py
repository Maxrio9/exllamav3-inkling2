"""CPU checks for the official Inkling template and provisional recipes."""
import json
import os
from pathlib import Path

import jinja2
from jinja2.ext import Extension
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = next(
    (tabby / "tests/fixtures/inkling2.jinja"
     for tabby in (Path(os.environ.get("INKLING_TABBY", ROOT / "tabby")), ROOT.parent / "tabby")
     if (tabby / "tests/fixtures/inkling2.jinja").is_file()),
    ROOT / "tabby/tests/fixtures/inkling2.jinja",
)


class GenerationExtension(Extension):
    tags = {"generation"}

    def parse(self, parser):
        token = next(parser.stream)
        body = parser.parse_statements(["name:endgeneration"], drop_needle=True)
        return jinja2.nodes.Scope(body).set_lineno(token.lineno)


def render(**kwargs):
    if not TEMPLATE.is_file():
        raise FileNotFoundError(f"Official Inkling template missing: {TEMPLATE}")
    env = jinja2.Environment(extensions=[GenerationExtension])
    env.globals["raise_exception"] = lambda message: (_ for _ in ()).throw(ValueError(message))
    env.filters["tojson"] = lambda value, **kw: json.dumps(value, ensure_ascii=False, **kw)
    return env.from_string(TEMPLATE.read_text()).render(**kwargs)


def test_template_multiturn_reasoning_and_tools():
    messages = [
        {"role": "user", "content": "Weather?"},
        {"role": "assistant", "content": None, "reasoning_content": "Check it.",
         "tool_calls": [{"id": "call1", "function": {"name": "weather", "arguments": {"city": "Berlin"}}}]},
        {"role": "tool", "tool_call_id": "call1", "content": "sunny"},
    ]
    result = render(messages=messages, tools=[{"type": "function", "function": {"name": "weather", "parameters": {}}}], reasoning_effort="low")
    assert "Thinking effort level: 0.2" in result
    assert "<|content_thinking|>Check it.<|end_message|>" in result
    assert '<|content_invoke_tool_json|>{"name":"weather","args":{"city":"Berlin"}}' in result
    assert "<|message_tool|>weather<|content_text|>sunny" in result


def test_template_default_and_bad_effort():
    assert "Thinking effort level: 0.9" in render(messages=[], tools=[])
    with pytest.raises(ValueError, match="Unknown reasoning_effort"):
        render(messages=[], tools=[], reasoning_effort="extreme")


@pytest.mark.parametrize("name", ["tabby-16gb-3.3bpw.yml", "tabby-24gb-4.0bpw.yml", "tabby-24gb-4.75bpw.yml"])
def test_draft_recipes(name):
    recipe = yaml.safe_load((ROOT / "configs" / name).read_text())
    assert recipe["model"]["cache_mode"] == ("Q8" if "16gb" in name else "FP16")
    assert recipe["model"]["warmup"] is True
    assert recipe["model"]["recurrent_checkpoint_interval_pp"] == recipe["model"]["chunk_size"]
    assert recipe["model"]["max_batch_size"] == 1
    assert recipe["model"]["template_vars_default"]["reasoning_effort"] == "high"
    assert recipe["draft_model"]["draft_mode"] == "disabled"
    variant = ROOT / "configs" / name.replace(".yml", "-dspark.yml")
    recipe["draft_model"]["draft_mode"] = "mtp"
    recipe["model"]["cache_mode"] = "FP16"
    assert yaml.safe_load(variant.read_text()) == recipe
