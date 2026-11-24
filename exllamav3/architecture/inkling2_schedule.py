"""Inkling 2 visit schedule and loop builder (no package-relative imports, so tests can load it by path)."""
from dataclasses import dataclass

@dataclass(frozen=True)
class InklingVisit:
    index: int
    layer: int
    pass_index: int
    cache_slot: int


def inkling2_visit_schedule(num_layers=24, core_start=6, core_end=18, passes=3):
    """Return execution order with PR-compatible physical cache slots."""
    if not (0 <= core_start < core_end <= num_layers and passes >= 1):
        raise ValueError("invalid Inkling loop bounds or pass count")
    visits = []
    def add(layer, pass_index, slot):
        visits.append(InklingVisit(len(visits), layer, pass_index, slot))
    for layer in range(core_start):
        add(layer, 0, layer)
    for p in range(passes):
        for layer in range(core_start, core_end):
            slot = layer if p == 0 else num_layers + (p - 1) * (core_end - core_start) + layer - core_start
            add(layer, p, slot)
    for layer in range(core_end, num_layers):
        add(layer, 0, layer)
    return tuple(visits)


def inkling2_schedule_from_config(raw):
    """Adapt slice 2's raw configuration to the visit/state execution plan.

    This independent section can coexist with slice 2's visit_plan and config.
    """
    text = raw['text_config']
    loop = text['loop_config']
    values = (text['num_hidden_layers'], loop['core_start'], loop['core_end'], loop['num_passes'])
    if any(type(value) is not int for value in values) or loop.get('is_varloop', False):
        raise ValueError('Invalid Inkling loop configuration')
    return inkling2_visit_schedule(*values)


def build_inkling2_loop(raw, layers, lm_head, loop_class):
    """Build execution from slice 2's raw config and supplied physical layers."""
    text = raw['text_config']
    loop = text['loop_config']
    schedule = inkling2_schedule_from_config(raw)
    if len(layers) != text['num_hidden_layers']:
        raise ValueError('physical layer count does not match config')
    return loop_class(layers, schedule, loop['core_start'], loop['core_end'],
                      loop['num_passes'], text['hidden_size'], lm_head,
                      eps=text.get('rms_norm_eps', 1e-6),
                      softcap=text.get('final_logit_softcapping', 30))
