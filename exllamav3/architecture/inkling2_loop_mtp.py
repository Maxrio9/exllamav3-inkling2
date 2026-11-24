"""Configuration validation for the Inkling2 DSpark draft."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Inkling2MTPConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    local_layer_ids: tuple
    sliding_window_size: int
    block_size: int
    markov_rank: int
    target_layer_ids: tuple
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int

    @classmethod
    def from_target(cls, config):
        mtp, text = config['mtp_config'], config['text_config']
        loop = text['loop_config']
        visits = text['num_hidden_layers'] + (loop['num_passes'] - 1) * (loop['core_end'] - loop['core_start'])
        ids = tuple(mtp['target_layer_ids'])
        if (len(ids) < 2 or tuple(sorted(set(ids))) != ids or ids[0] < 0 or ids[-1] != visits
                or mtp['hidden_size'] != text['hidden_size'] or mtp['markov_head_type'] != 'rnn_hidden'
                or mtp.get('q_bias', False) or mtp.get('o_bias', False)
                or loop.get('is_varloop', False)):
            raise ValueError('unsupported Inkling2 DSpark configuration')
        fields = cls.__dataclass_fields__
        args = {name: mtp[name] for name in fields}
        args['local_layer_ids'] = tuple(args['local_layer_ids'])
        args['target_layer_ids'] = ids
        if (args['block_size'] < 2 or args['num_hidden_layers'] < 1 or args['head_dim'] % 2
                or args['num_key_value_heads'] < 1 or args['num_attention_heads'] % args['num_key_value_heads']
                or any(i < 0 or i >= args['num_hidden_layers'] for i in args['local_layer_ids'])):
            raise ValueError('invalid Inkling2 DSpark dimensions')
        return cls(**args)
