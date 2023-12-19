import torch
import warnings 
import transformers
from torch import nn
from .model import SBFEncoder, SBFEncoderDecoderModel

warnings.filterwarnings('ignore') 

def make_model_and_tokenizer(config):
    if config['name'] == 'gpt2':
        return _make_gpt2_model_and_tokenizer(config)
    elif config['name'] == 'enc_dec':
        return _make_enc_dec_model_and_tokenizer(config)
    
    
def _make_gpt2_model_and_tokenizer(config):
    tokenizer = _make_gpt2_tokenizer(config)
    model = _make_gpt2_model(config)
    return model, tokenizer


def _make_enc_dec_model_and_tokenizer(config):
    tokenizer = transformers.AutoTokenizer.from_pretrained(config['checkpoint_name'])
    model = _make_encoder_decoder_model(tokenizer, config)
    return model, tokenizer


def _make_gpt2_tokenizer(config:dict, verbose=False):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config['checkpoint_name'],
        padding_side=config['padding_side'],
        use_fast=True
    )
    tokenizer.add_special_tokens(config['special_tokens'])
    if verbose:
        print("List of all special token and its token_id:")
        print(" -", tokenizer.all_special_tokens)
        print(" -",tokenizer(tokenizer.all_special_tokens)["input_ids"])

    return tokenizer


def _make_gpt2_model(tokenizer, config):
    model = transformers.GPT2LMHeadModel.from_pretrained(config['checkpoint_name'])
    # init new embedding
    new_tokens = len(tokenizer) - model.config.vocab_size
    model.resize_token_embeddings(len(tokenizer))
    model = _init_gpt2_new_tokens_embs(model, new_tokens)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.sep_token_id = tokenizer.sep_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    return model


def _make_encoder_decoder_model(tokenizer, config, initialize_cross_attention=True):
    encoder = transformers.AutoModel.from_pretrained(config['checkpoint_name'])
    encoder = SBFEncoder(encoder, encoder.config)

    decoder = transformers.AutoModelForCausalLM.from_pretrained(
        config['checkpoint_name'],
        is_decoder=True,
        add_cross_attention=True,
        decoder_start_token_id=tokenizer.bos_token_id,
    )

    config = transformers.EncoderDecoderConfig.from_encoder_decoder_configs(
        encoder.config,
        decoder.config,
        tie_encoder_decoder=True,
        decoder_start_token_id=tokenizer.cls_token_id,
        eos_token_id=tokenizer.sep_token_id,
        pad_token_id=tokenizer.pad_token_id,
        # sensible parameters for generation
        vocab_size=decoder.config.vocab_size,
        max_new_tokens=config['decoder_max_length'],
        **config['generation_parameters'],
    )

    model = SBFEncoderDecoderModel(encoder=encoder, decoder=decoder, config=config)
    if initialize_cross_attention:
        _initialize_cross_attention_with_self_attention(model)

    return model

def _init_gpt2_new_tokens_embs(model, new_tokens):
    params = model.state_dict()
    embeddings = params['transformer.wte.weight']
    pre_expansion_embeddings = embeddings[:-new_tokens,:]
    mu = torch.mean(pre_expansion_embeddings, dim=0)
    n = pre_expansion_embeddings.size()[0]
    sigma = ((pre_expansion_embeddings - mu).T @ (pre_expansion_embeddings - mu)) / n
    dist = torch.distributions.multivariate_normal.MultivariateNormal(mu, covariance_matrix=1e-5*sigma)
    pad_embedding = (torch.ones_like(mu)*torch.finfo(torch.float16).min).unsqueeze(0) # (1, 768)
    other_embes = torch.stack(tuple(dist.sample() for _ in range(new_tokens-1)), dim=0) # (11,768)
    new_embeddings = torch.cat((pad_embedding,other_embes), dim=0) # (12, 768)
    # new_embeddings = torch.stack(tuple((dist.sample() for _ in range(new_tokens))), dim=0)
    embeddings[-new_tokens:,:] = new_embeddings
    params['transformer.wte.weight'][-new_tokens:,:] = new_embeddings
    model.load_state_dict(params)
    return model


def _initialize_cross_attention_layer_with_self_attention_layer(
    self_attention: nn.Module,
    cross_attention: nn.Module,
    cross_attention_layer_prefix: str,
):
    uninitialized_cross_attention_weights: list = []
    if cross_attention.__class__ != self_attention.__class__:
        print(
            f"{cross_attention.__class__} and {self_attention.__class__} are not equal. In this case make sure that all encoder"
            " weights are correctly initialized."
        )

    def _initialize_cross_attention_with_self_attention_recursively(
        self_attention_pointer: nn.Module,
        cross_attention_pointer: nn.Module,
        module_name: str,
        uninitialized_cross_attention_weights:list,
        depth=0,
    ):
        assert isinstance(self_attention_pointer, nn.Module) and isinstance(
            cross_attention_pointer, nn.Module
        ), f"{self_attention_pointer} and {cross_attention_pointer} have to be of type nn.Module"
        if hasattr(self_attention_pointer, "weight"):
            assert hasattr(cross_attention_pointer, "weight")
            cross_attention_pointer.weight.data = (
                self_attention_pointer.weight.data.clone().detach()
            )
            if hasattr(self_attention_pointer, "bias"):
                assert hasattr(cross_attention_pointer, "bias")
                cross_attention_pointer.bias.data = (
                    self_attention_pointer.bias.data.clone().detach()
                )
            return

        cross_attention_modules = cross_attention_pointer._modules
        self_attention_modules = self_attention_pointer._modules
        if len(self_attention_modules) > 0:
            assert (
                len(cross_attention_modules) > 0
            ), f"Cross-attention module {cross_attention_pointer} does not match self-attention module {self_attention_pointer}"

            all_cross_attention_weights = {
                module_name + "/" + sub_name
                for sub_name in cross_attention_modules.keys()
            }
            cross_attention_layer_pos = 0
            for name, module in self_attention_modules.items():
                if name.isdigit():
                    cross_attention_name = str(int(name) + cross_attention_layer_pos)
                    self_attention_name = name
                    if not isinstance(
                        self_attention_modules[self_attention_name],
                        type(cross_attention_modules[cross_attention_name]),
                    ) and len(cross_attention_modules) != len(self_attention_modules):
                        # this can happen if the name corresponds to the position in a list module list of layers
                        # in this case the decoder has added a cross-attention that the encoder does not have
                        # thus skip this step and subtract one layer pos from encoder
                        cross_attention_layer_pos -= 1
                        continue
                elif name not in cross_attention_modules:
                    continue
                elif depth > 500:
                    raise ValueError(
                        "Max depth of recursive function `initialize_cross_attention_with_self_attention` reached. It seems that there is"
                        " a circular dependency between two or more `nn.Modules` of your model."
                    )
                else:
                    self_attention_name = cross_attention_name = name
                _initialize_cross_attention_with_self_attention_recursively(
                    self_attention_modules[self_attention_name],
                    cross_attention_modules[cross_attention_name],
                    module_name + "/" + name,
                    uninitialized_cross_attention_weights,
                    depth=depth + 1,
                )
                all_cross_attention_weights.remove(
                    module_name + "/" + cross_attention_name
                )

            uninitialized_cross_attention_weights += list(all_cross_attention_weights)

    # initialize weights recursively
    _initialize_cross_attention_with_self_attention_recursively(
        self_attention,
        cross_attention,
        cross_attention_layer_prefix,
        uninitialized_cross_attention_weights,
    )
    if len(uninitialized_cross_attention_weights) > 0:
        warnings.warn(
            f"The following cross_attention weights were not initialized with self_attention weights: {uninitialized_cross_attention_weights}"
        )


def _initialize_cross_attention_with_self_attention(model):
    decoder_base_model_prefix = model.decoder.base_model_prefix
    for layer_idx in range(model.config.decoder.num_hidden_layers):
        decoder_layer = model.decoder._modules[decoder_base_model_prefix].encoder.layer[
            layer_idx
        ]
        cross_attention = decoder_layer.crossattention
        self_attention = decoder_layer.attention
        cross_attention_name = f"layer.{layer_idx}.crossattention"
        _initialize_cross_attention_layer_with_self_attention_layer(
            self_attention, cross_attention, cross_attention_name
        )
    print("Cross-attention has been initialized with self-attention weights.")