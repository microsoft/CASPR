import onnx
import torch
from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions, get_available_providers
from torch.onnx import register_custom_op_symbolic

from caspr.models.factory import LSTM, TRANSFORMER
from caspr.utils.preprocess import get_nonempty_tensors
from caspr.utils.score import get_architecture

OPSET_VERSION = 12
SEQ_CAT_INDEX = 0
SEQ_CONT_INDEX = 1
NON_SEQ_CAT_INDEX = 2
NON_SEQ_CONT_INDEX = 3

_onnx_opset_version = 1

def register_custom_op():
    """
    This function registers symbolic functions for
    custom ops that are implemented as part of ONNX Runtime
    """

    # Symbolic definition
    def inverse(g, self):
        return g.op("com.microsoft::Inverse", self)

    def gelu(g, self):
        return g.op("com.microsoft::Gelu", self)

    def triu(g, self, diagonal):
        return g.op("com.microsoft::Trilu", self, diagonal, upper_i=1)

    def tril(g, self, diagonal):
        return g.op("com.microsoft::Trilu", self, diagonal, upper_i=0)

    # Op Registration
    register_custom_op_symbolic('::inverse', inverse, _onnx_opset_version)
    register_custom_op_symbolic('::gelu', gelu, _onnx_opset_version)
    register_custom_op_symbolic('::triu', triu, _onnx_opset_version)
    register_custom_op_symbolic('::tril', tril, _onnx_opset_version)


def unregister_custom_op():
    """
    This function unregisters symbolic functions for
    custom ops that are implemented as part of ONNX Runtime
    """

    import torch.onnx.symbolic_registry as sym_registry

    # TODO: replace this once PyTorch supports unregister natively.
    def unregister(name, opset_version):
        ns, kind = name.split("::")
        from torch.onnx.symbolic_helper import _onnx_stable_opsets

        for version in _onnx_stable_opsets:
            if version >= opset_version and sym_registry.is_registered_op(kind, ns, version):
                del sym_registry._registry[(ns, version)][kind]

    unregister('::inverse', _onnx_opset_version)
    unregister('::gelu', _onnx_opset_version)
    unregister('::triu', _onnx_opset_version)
    unregister('::tril', _onnx_opset_version)


def get_input_names(nonempty_idx):
    mapping = {SEQ_CAT_INDEX: 'seq_cat', SEQ_CONT_INDEX: 'seq_cont',
               NON_SEQ_CAT_INDEX: 'non_seq_cat', NON_SEQ_CONT_INDEX: 'non_seq_cont'}
    input_names = [mapping[idx] for idx in nonempty_idx if idx in mapping] + ['nonempty_idx']
    return input_names


def get_dummy_inputs(model):
    if get_architecture(model) == TRANSFORMER:
        seq_cat_dim = len(model.unified_encoder.emb_seq.emb_layers)
        seq_cont_dim = model.unified_encoder.seq_cont_dim
        non_seq_cat_dim = len(model.unified_encoder.emb_non_seq.emb_layers)
        non_seq_cont_dim = model.unified_encoder.non_seq_cont_dim
        adjust_seq_len = model.unified_encoder.transformer_encoder.pos_embedding.num_embeddings
        seq_len = adjust_seq_len - int((non_seq_cat_dim + non_seq_cont_dim) > 0)
    elif get_architecture(model) == LSTM:
        seq_cat_dim = model.unified_encoder.seq_cat_count
        seq_cont_dim = model.unified_encoder.seq_cont_count
        non_seq_cat_dim = model.unified_encoder.non_seq_cat_count
        non_seq_cont_dim = model.unified_encoder.non_seq_cont_count
        seq_len = model.unified_encoder.seq_len

    device = next(model.parameters()).device
    seq_cat_dummy = torch.zeros((1, seq_len, seq_cat_dim), dtype=torch.long, device=device)
    seq_cont_dummy = torch.zeros((1, seq_len, seq_cont_dim), dtype=torch.float32, device=device)
    non_seq_cat_dummy = torch.zeros((1, non_seq_cat_dim), dtype=torch.long, device=device)
    non_seq_cont_dummy = torch.zeros((1, non_seq_cont_dim), dtype=torch.float32, device=device)

    dummy = (seq_cat_dummy, seq_cont_dummy, non_seq_cat_dummy, non_seq_cont_dummy)
    nonempty_tensors, nonempty_idx = get_nonempty_tensors(dummy)
    dummy_inputs = (*nonempty_tensors, torch.tensor(nonempty_idx))

    input_names = get_input_names(nonempty_idx)

    return dummy_inputs, input_names


def export_onnx(model, model_path):
    model.eval()

    dummy_inputs, input_names = get_dummy_inputs(model)

    with torch.no_grad():
        dummy_outputs = model.unified_encoder(*dummy_inputs)
    output_names = [f"output_{i}" for i in range(len(dummy_outputs))]

    dynamic_axes = dict.fromkeys(input_names + output_names, {0: 'batch_size'})
    torch.onnx.export(model=model.unified_encoder,
                      args=dummy_inputs,
                      f=model_path,
                      input_names=input_names,
                      output_names=output_names,
                      dynamic_axes=dynamic_axes,
                      opset_version=OPSET_VERSION,
                      custom_opsets={'com.microsoft': 1},
                      do_constant_folding=True)


def to_numpy(tensor):
    if tensor.requires_grad:
        return tensor.detach().cpu().numpy()
    else:
        return tensor.cpu().numpy()


class ONNXWrapper:

    def __init__(self, model_path_or_proto, model_type=TRANSFORMER):
        if isinstance(model_path_or_proto, str):
            with open(model_path_or_proto, 'rb') as model_file:
                self.model_bytes = model_file.read()
        else:
            self.model_bytes = onnx._serialize(model_path_or_proto)
        self.session = self.load()
        self.model_type = model_type

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['session']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.session = self.load()

    def unified_encoder(self, *args):
        nonempty_tensors = args[:-1]
        inputs = list(map(to_numpy, nonempty_tensors))
        ort_inputs = dict((self.session.get_inputs()[i].name, inp) for i, inp in enumerate(inputs))
        return (torch.from_numpy(out) for out in self.session.run(None, ort_inputs))

    def load(self, device=torch.device('cpu'), enable_all_optimization=True):
        sess_options = SessionOptions()
        sess_options.graph_optimization_level = (
            GraphOptimizationLevel.ORT_ENABLE_ALL
            if enable_all_optimization
            else GraphOptimizationLevel.ORT_ENABLE_BASIC
        )

        use_gpu = 'cuda' in device.type and 'CUDAExecutionProvider' in get_available_providers()
        execution_providers = (
            ["CPUExecutionProvider"] if not use_gpu else ["CUDAExecutionProvider", "CPUExecutionProvider"]
        )

        session = InferenceSession(self.model_bytes, sess_options, providers=execution_providers)
        return session

    def to(self, device):
        self.session = self.load(device)

    def cpu(self):
        self.to(torch.device('cpu'))

    def eval(self):
        pass
