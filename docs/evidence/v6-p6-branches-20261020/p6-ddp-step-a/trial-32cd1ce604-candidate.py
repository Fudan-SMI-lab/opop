"""Joint replacement bundle for the selected GPT-2 block compute path.

Scope: the 21 selected groups (GPT2Block, GPT2Attention, GPT2MLP,
NewGELUActivation, LayerNorm, p=0 Dropout, the four Conv1D families) across
both accumulation phases on every rank, exactly the joint selection.

Replacements are numerically-equivalent eager compositions of the observed
math:

* NewGELUActivation's multi-kernel elementwise chain
  (pow -> mul -> add -> mul -> tanh -> add -> mul -> mul) becomes the single
  fused aten::gelu(approximate="tanh") kernel plus its single-kernel
  autograd counterpart. The formula implemented by that kernel is identical:
  0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3))).
* Dropout with p == 0 becomes an exact identity: aten native_dropout(p=0)
  already returns its input alias, so this only removes dispatch overhead and
  keeps aliasing/autograd semantics bit-identical.
* LayerNorm and Conv1D keep their exact aten ops (native_layer_norm / addmm)
  with lean argument handling and cached parameter references (parameters are
  updated in place by SGD and reset via load_state_dict, so the tensor
  objects stay valid for the whole warm-model lifecycle).
* GPT2Attention keeps the eager attention math (bmm q@k^T, additive causal
  mask, softmax, bmm probs@v) but folds the exact power-of-two 1/sqrt(64)
  scaling and the additive mask into one torch.baddbmm epilogue
  (alpha=scaling, beta=1), replacing matmul+mul+add with a single GEMM and
  removing the per-call attention-interface registry lookup.
* GPT2Block/GPT2MLP re-express the same submodule calls (ln_1 -> attn ->
  residual -> ln_2 -> mlp -> residual) so the framework's per-site
  replacements for the inner modules still fire and stay observable.

DDP reducer order, bucket layout, collective participation, RNG consumption,
external outputs, gradients and parameter updates are untouched. Any
non-observed signature, kwarg combination or dtype falls back to the captured
parent forward.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

_SEMANTICS_BLOCK = "transformers.models.gpt2.modeling_gpt2.GPT2Block"
_SEMANTICS_ATTENTION = "transformers.models.gpt2.modeling_gpt2.GPT2Attention"
_SEMANTICS_MLP = "transformers.models.gpt2.modeling_gpt2.GPT2MLP"
_SEMANTICS_GELU = "transformers.activations.NewGELUActivation"
_SEMANTICS_LAYERNORM = "torch.nn.modules.normalization.LayerNorm"
_SEMANTICS_DROPOUT = "torch.nn.modules.dropout.Dropout"
_SEMANTICS_CONV1D = "transformers.pytorch_utils.Conv1D"

# Exact call IDs of the joint selection (union of the 21 selected groups).
_SELECTED_CALLS = (
    "000efc6fd26ad966409551aa", "00a9b01e66b293d08e043c03", "00ae3847dd51a38c1add902e", "00b3062208bc4bfe0c01e050",
    "02b5af6c699fda4db8a01746", "03b91429291b948e4d77cd03", "0595a64a2d7dcd371afe61f6", "0662ef1307a0b9ff0e79cae8",
    "076b7f5b1e0e178567dc108a", "08025c00573b84c9117bbc2f", "08933daac7e45443f4bc0172", "0924fa92c531d02254b2e51b",
    "0979a799888d2cd5be857727", "0b65798d4404a29db3853482", "0c8f50360c38018aad10c178", "0c9fa15507c4045457061932",
    "0ed985c80e6073981554e2f3", "0f203bff432f4e91c50de643", "0f5f0e05a3fbf3f4ce5059ab", "109688cc786795ea487db536",
    "10ae9eafdd271accd7e0dbbb", "10dda789c452f897f1a408f9", "12720b5cee6eb676b8379894", "14e655ec95a7cbb9c1225b36",
    "15344599206e35d185669634", "16614833389581cad3762aff", "16aecb6fb5dfaf9dc2213bb1", "16d3b6f847f4088b9487b855",
    "189bb295279c420e0d8e8f11", "19e7453f2c247349c20d1562", "1a71e9196f96fcf6524e0e4d", "1d973b145f84ef2d97387273",
    "1da910dc9a736ad03dc78f8c", "1db76f17bf81cbe1b69e0e4b", "1e34176ba03d9ccf3f0b5297", "1f19898e1e8084dcf77d00a1",
    "20080e42dbbf87fb2b7e7997", "205bf860d36a5ea2eb11fe98", "21ae1c50bdf7012d6008d024", "21db6114783b54d16ccf764f",
    "23a7b6eeb169e3449595cb90", "2447e4bfde89d4eea7dd92e4", "24752b614beb5d044d7e7fbd", "248a03afe70cb51d2ce88ee0",
    "24b47f7fcba444a42fa0111e", "2564366062c2f8e130b09313", "26910a3a04f67dea7f6fa2ab", "26fd63b01e68fd81ae5bd049",
    "2765d1d7d9c768be0ef581d7", "27a5f56f2aa51396887f3b89", "2959cf4ac6fb9558a15b96a3", "2973576612144284be11dc76",
    "2b944be4495dc6ae9f1d7557", "2dc1e97aa668767349c134cf", "2e23899b05fea2eb85f6e0ca", "2e5abe37c1b2f8cde43bb6dd",
    "2f3eb519eaa37ecf44f3c79f", "2fb1f11c1b0d050511f08434", "2fec3550d645d0a6b0d8c8c1", "30cadf3f6cc5580731a38a99",
    "3239d25a9e07cf4c1546941b", "3320879937b70dcd76c4e0da", "340d9df8e0e0bab04952d0b7", "343f54cd0543646846cb2445",
    "34b1ca4eb587e4efa857663f", "35ea2a023a9204d7cdbfcfc4", "37025c4b237f44ab01498216", "3811658e2f991c843a6c879c",
    "3864791b91c4e1069b5124be", "39212b9575691ed2bfeab244", "3a56ca90186c8574c01ba4cd", "3b1f58153b299a3ff68c3c11",
    "3d5e4c73dc0ea4572bc7a713", "3e205d433117f25d3f2a6cd6", "3fd35073ba572d645c527bf1", "4125775d5c742e572337313e",
    "41dab85be2151477b1b961e1", "428b6ca8d2d4ead0847f2936", "429113cf07debbea382ee94a", "42fc8d853123ee7fe426b5f5",
    "477de62fa7ced2cabc92867f", "489f0ed59618ae0785aa0543", "497b7707e18aa0e5a0a58c6e", "499f36c49a2ce4639701e5ad",
    "4a310cb655f42330a0a46a73", "4bc0301b1971c86ccdb51804", "4cf9021f7010efed1a1b014f", "4da2cfefaad119dbcced9342",
    "4de8e58424e7ade2f58d853d", "4ec2198597eed2ba805d4099", "5038514c760dbec126fa663c", "5053bedb6c9523e3cf7c86c9",
    "523ad2cdc643bbb855876eab", "533b4b3ee773f1eb5d0004b1", "54384cde284bd37d7ebb976b", "55c03115843e16089537c6bd",
    "55c06b990612d30401eb4e0d", "56e540b20cc1a904307b6ee3", "57f73313975175097c6d4a4a", "59f46b358068be010745183d",
    "5a2e24bb5591e541ca9682c6", "5b465d1774bddf829045486e", "5b803d37942d1b1674afa061", "5c2e6a983b2c9d7c0b92a0f4",
    "5c6641a3eee83a6f8d4ec48e", "5de8fcea660e4c38dac48056", "5e0d18196ce9368b6919f8cb", "61f0c8254c43c3bb5f33260f",
    "631e12cbad7d4bec3945c0e7", "63c9ed914e22accf398afda8", "64ab481adbf916d8350f4583", "66cb15314c9479ca90adcd78",
    "670d3cd2e2cd1754a2f20815", "67a8cbc62dcd0c25a1acdc58", "68790ff99463f221211b0fc5", "68a3d3889b6c906aa897e7c8",
    "68d81dba9c5679a6be3ee370", "6963011540116c9aabf75615", "6a784b3532c714fab9dea9ca", "6b7dd0abb8b551096c6db7bc",
    "6c05309ca54460fd4d6821fd", "6d3fc957255fa743ab38d6c7", "6d8b274e620e4a1e2e3a69b7", "6e90fda1779ea03762eefb9b",
    "7270c2d977cc66f6629fa420", "727c1707edcc3eb87d6acdbe", "7296f1f7f49eb1dddaa52484", "730b30fb2b7ba8d3434c5c1a",
    "733f1e03d8916cbdc417b45e", "7452bce93a0707d0a13781ef", "74b198a1b30ad8645d2be803", "754d250e7096d61e09c17989",
    "760e2e70983c353a97c849a2", "76238a04556aa7822b704ab4", "7768f15f291c34217fe14ef6", "7772d67fda89c707687f123d",
    "788ff8147628e229cdb583a8", "7a90132395c5ec32ec989720", "7aba96c69c66cbbfb6ccc16b", "7aefa266c67d10fda0f08f78",
    "7cf2f580c894c07cf5a4d90d", "83105e9763f9f65448b77167", "839ab22a494e6da2dbbb2080", "840bcc2a62cb3e419f7ebc52",
    "844ed5d59fdc2e085ca0f549", "85443ca48a50fdced6743b67", "86c1044ca74afde88d17474d", "87800eae12fded83bddbf14f",
    "879d1fcf2133c55fc5bbbb42", "884b997e287d7dbc495b2886", "89226389985e1a15d9ad6c27", "89e345d9cddc4b2e3d771a61",
    "8b81cac0a46f729fa6a037ed", "8b9715a136064a4e21eef4a5", "8c4028077c0372aec1b96c7d", "8ce79700833e7ebd27eda110",
    "8d0d4e394f88c300b23aeb6e", "8e540b3f20afde65a1b845b9", "8e5ecc8ff499770242b4ae5a", "8ea5871af28842800a42659c",
    "8eaf654003c062bc209466bc", "9031e4ad3d6ad1524b9f08f7", "9103bee2490c8917ea585711", "927caef13d63c4e3443ebb90",
    "9382b87014161fb7b03882f6", "939b889669f9149f70512516", "93bf0090f25136392448fb63", "941a47289925a877bfd7366d",
    "946f41c0b5211f364bed04fd", "956bb90629a3cf4b150db94e", "96b03dff9fe2b127187e8c34", "984cd6a5818ecbd326594341",
    "9ae676552278a8b5e4113ec1", "9af71a3185dd0f0df197321c", "9b2a483e116762beaa48024e", "9c1fce0dbd29ab07b8d4e17e",
    "9e14204ecb0f1337d89ffc71", "9fcbcea15e86d89fc041021e", "a248132d788393cef339c7a4", "a3561e40dbba47fd1f6645aa",
    "a405d13311aa74ce7f4e4dec", "a42d5d714c462ee1bb555516", "a46ef68f9c3fed5e15b278da", "a4bb9b3b0dfa9909873bf251",
    "a5469bf093e76ef75ce7858f", "a5b41f97f86ba2b903aba2ab", "a6bd714aa42cdf53ece46db6", "a7003d941eabd3fab7a17135",
    "a708e495c22fc152c0552a88", "a925c6b20e0f8b3d0c2498f5", "aa4e5fa8c7f147ceee18f2b6", "aaf77edd97468d35b5a8bee6",
    "adf35bb10d531dfa6e8d464e", "ae36c4ae9977c20d332aa112", "b0e7b07316b62d075672f036", "b1301d22d8feb64fdadb2c8f",
    "b1a73d99f63bd88579ce1c45", "b1dbe90e75ae27f8903791eb", "b2e2a7afbf28f8954195701e", "b3990efa245d602a3d4ba647",
    "b3ef57b7ea9d53cb3e97ea4a", "b62872358255fd1657c85246", "b6f015a42f4836fbd0904703", "b71e8e914b91d47cfe1750c3",
    "b737de9ae72d1f4c1f2ca53c", "b754838031d9957a741927ac", "b7740714ee9ebf994c306fcc", "b7c028bdac89c581fccb9c2a",
    "b8bbd0f64994688fe92fbfce", "ba532c2a0b1b1ba4d5bb6714", "bb0aa8b079ddb34796506ec0", "bb10599cc6eb36096048eca0",
    "bd1f33ee5728f145ee8c2c97", "bd4ccd45e8d2daa142d8c37c", "beec2ff5992c5882f57d0c19", "bf086167121b4aa5089372e1",
    "c00a4215923a66fb35f98dd6", "c0d05d923936d6ad7854919b", "c1010b20ad59e240c8e5cbf2", "c1c56b2295dce7db9e9cb4e3",
    "c2dcff3f2565d5624a95a8a3", "c30fb5b6971854bd3460089e", "c3c0c18ce66ea4895be81558", "c61930e4f07489a519200285",
    "c6be5101fbe7d3534c42829d", "c7574855a80f3c1b2f9dbd56", "c794af516e41321c2e25fa6c", "c79e495fae70e0e593949ade",
    "c887cb8596d5d6f54fa91117", "c8d58507540983861d0e64df", "c8dc0b89256a6a0e809f4af8", "c996fdc14b469ad5ac59a510",
    "ca9d0514de6be0f1ade0754e", "cad5747ac3e09260a6d1a60b", "cb2752c159d367e8f4a2361c", "cbd981410ad87bb1d0ca6bbc",
    "cdbeb7173720b8602fa29774", "ce7360f2d8c79f18eeb3cd61", "ced29882c1c3d7d6cbc0035c", "cf7fd82f5af4b52cf68b9fee",
    "d1815457b719d3584b0f3a6d", "d46755daa1525ebb4767c6a9", "d4a3f81b520e5ee9a21436cf", "d52d8ea542eee5130c9000b7",
    "d5ee0b0bbe954901d172dd64", "d60bb99542c283a96399f381", "d6cc1b299f07e8e78d0aa246", "d8a39489fa74ed520950ac3e",
    "d8a5398ed4a479787074e400", "d8f8aef01b4be0cbe028f33b", "d9044ab4d613ceb212a35f4f", "da658176b352a21d85eb76b7",
    "da938e19399226e33361e552", "dbd24a933bb62a6153968c17", "dbe2dcf7e9298c9aeead764f", "dd7a911dfe7e96c0c7460df4",
    "ded68bfe86a4c165663b12a7", "df5843dbe251d620675e2309", "dfa773cc719e4452e48fd21f", "dfe1eadb5a7261f373a534e2",
    "e06c282578143e9d930c44d6", "e19e54c9a06301140e14b5d7", "e1e171a8c99d81663a13a09d", "e21bc0796fc18f9e726caca0",
    "e34a4138e923e12832b8ee75", "e382c545bbfa02155bd404b8", "e39fc9bd9dbfe548ea49e02b", "e3cb550e7efc0bf70177560c",
    "e4077e6bdfa7c9e8717bb281", "e4593e5ba12cf65473124994", "e4d823ee5976a2ca69deeebb", "e585698fb26e38037bda2cfa",
    "e5c31ed4d7cfd86e30748704", "e82f8283a6221a76536b0076", "e9b60bd7064f917ae1691256", "ea9f773bfba4fa4355a0dac9",
    "ebd55f83efe9f6e53da141de", "ed688899ec20f87484b18f23", "ee0f0d27536da0df1cb9ad56", "eedb65b89d1b2aed9abffa8c",
    "ef03c8cc6719a4a9125cb881", "efbb3ba13d5c9ae0bbac07ee", "f3118cb1d2b5519bf829a518", "f490e72d3b7269b0d51db929",
    "f58a45129c5a046e30e87817", "f5c285742f850b2135d70421", "f6284e97ce0a1a487f65142c", "f7ca80603032204c0dd2ac9f",
    "fc1beb382c850de39bde8e2a", "fd2582ddb7286c5f573a7c14", "fe25c108720a43dcbc54501f", "fea3699a870dfbd059ffec18"
)

_SELECTED = frozenset(_SELECTED_CALLS)


def _make_block(module, parent):
    ln_1 = module.ln_1
    attn = module.attn
    ln_2 = module.ln_2
    mlp = module.mlp

    def forward(
        hidden_states,
        past_key_values=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        **kwargs,
    ):
        if (
            past_key_values is not None
            or encoder_hidden_states is not None
            or encoder_attention_mask is not None
            or use_cache
            or kwargs.get("output_attentions")
        ):
            return parent(
                hidden_states,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                **kwargs,
            )
        residual = hidden_states
        attn_output = attn(
            ln_1(hidden_states),
            past_key_values=None,
            attention_mask=attention_mask,
            use_cache=False,
            **kwargs,
        )[0]
        hidden_states = attn_output + residual
        residual = hidden_states
        hidden_states = residual + mlp(ln_2(hidden_states))
        return hidden_states

    return forward


def _make_attention(module, parent):
    if (
        getattr(module, "is_cross_attention", True)
        or getattr(module, "reorder_and_upcast_attn", True)
        or float(getattr(module.attn_dropout, "p", 1.0)) != 0.0
        or getattr(module.config, "_attn_implementation", "") != "eager"
    ):
        return parent

    c_attn = module.c_attn
    c_proj = module.c_proj
    resid_dropout = module.resid_dropout
    split_size = int(module.split_size)
    num_heads = int(module.num_heads)
    head_dim = int(module.head_dim)
    scaling = float(module.scaling)

    def forward(
        hidden_states,
        past_key_values=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        **kwargs,
    ):
        if (
            past_key_values is not None
            or encoder_hidden_states is not None
            or encoder_attention_mask is not None
            or output_attentions
            or hidden_states.dtype is not torch.float32
            or hidden_states.shape[0] != 1
        ):
            return parent(
                hidden_states,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
                **kwargs,
            )
        query, key, value = c_attn(hidden_states).split(split_size, dim=2)
        seq = query.shape[1]
        q = (
            query.view(1, seq, num_heads, head_dim)
            .transpose(1, 2)
            .reshape(num_heads, seq, head_dim)
        )
        k = (
            key.view(1, seq, num_heads, head_dim)
            .transpose(1, 2)
            .reshape(num_heads, seq, head_dim)
        )
        v = (
            value.view(1, seq, num_heads, head_dim)
            .transpose(1, 2)
            .reshape(num_heads, seq, head_dim)
        )
        if attention_mask is None:
            attn_weights = torch.bmm(q, k.transpose(-1, -2)) * scaling
        elif (
            attention_mask.dim() == 4
            and attention_mask.shape[0] == 1
            and attention_mask.shape[1] == 1
        ):
            attn_weights = torch.baddbmm(
                attention_mask[0, 0],
                q,
                k.transpose(-1, -2),
                beta=1.0,
                alpha=scaling,
            )
        else:
            return parent(
                hidden_states,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
                **kwargs,
            )
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.bmm(attn_weights, v)
        attn_output = attn_output.transpose(0, 1).reshape(1, seq, num_heads * head_dim)
        attn_output = c_proj(attn_output)
        attn_output = resid_dropout(attn_output)
        return attn_output, attn_weights.view(
            1, num_heads, attn_weights.shape[-2], attn_weights.shape[-1]
        )

    return forward


def _make_mlp(module, parent):
    c_fc = module.c_fc
    act = module.act
    c_proj = module.c_proj
    dropout = module.dropout

    def forward(hidden_states):
        hidden_states = c_fc(hidden_states)
        hidden_states = act(hidden_states)
        hidden_states = c_proj(hidden_states)
        return dropout(hidden_states)

    return forward


def _make_gelu(module, parent):
    def forward(input):
        if input.dtype is torch.float32:
            return F.gelu(input, approximate="tanh")
        return parent(input)

    return forward


def _make_layernorm(module, parent):
    weight = module.weight
    bias = module.bias
    eps = module.eps
    normalized_shape = tuple(module.normalized_shape)

    def forward(input):
        return F.layer_norm(input, normalized_shape, weight, bias, eps)

    return forward


def _make_dropout(module, parent):
    if float(module.p) != 0.0 or module.inplace:
        return parent

    def forward(input):
        return input

    return forward


def _make_conv1d(module, parent):
    weight = module.weight
    bias = module.bias
    nf = int(module.nf)

    def forward(x):
        size_out = x.size()[:-1] + (nf,)
        x = torch.addmm(bias, x.view(-1, x.size(-1)), weight)
        return x.view(size_out)

    return forward


_MAKERS = {
    _SEMANTICS_BLOCK: _make_block,
    _SEMANTICS_ATTENTION: _make_attention,
    _SEMANTICS_MLP: _make_mlp,
    _SEMANTICS_GELU: _make_gelu,
    _SEMANTICS_LAYERNORM: _make_layernorm,
    _SEMANTICS_DROPOUT: _make_dropout,
    _SEMANTICS_CONV1D: _make_conv1d,
}


def build(context):
    modules = context.get("modules") or {}
    references = context.get("references") or {}
    targets = context.get("targets") or {}

    replacements = {}
    for call_id in _SELECTED:
        spec = targets.get(call_id)
        module = modules.get(call_id)
        parent = references.get(call_id)
        if spec is None or module is None or parent is None:
            continue
        maker = _MAKERS.get(spec.get("semantics"))
        if maker is None:
            continue
        replacements[call_id] = maker(module, parent)
    return replacements
