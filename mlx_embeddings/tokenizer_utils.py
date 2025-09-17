import json
from functools import partial

import mlx.core as mx
from tokenizers import Tokenizer

REPLACEMENT_CHAR = "\ufffd"


def _remove_space(x):
    if x and x[0] == " ":
        return x[1:]
    return x


class StreamingDetokenizer:
    """The streaming detokenizer interface so that we can detokenize one token at a time.

    Example usage is as follows:

        detokenizer = ...

        # Reset the tokenizer state
        detokenizer.reset()

        for token in generate(...):
            detokenizer.add_token(token.item())

            # Contains the whole text so far. Some tokens may not be included
            # since it contains whole words usually.
            detokenizer.text

            # Contains the printable segment (usually a word) since the last
            # time it was accessed
            detokenizer.last_segment

            # Contains all the tokens added so far
            detokenizer.tokens

        # Make sure that we detokenize any remaining tokens
        detokenizer.finalize()

        # Now detokenizer.text should match tokenizer.decode(detokenizer.tokens)
    """

    __slots__ = ("text", "tokens", "offset")

    def reset(self):
        raise NotImplementedError()

    def add_token(self, token):
        raise NotImplementedError()

    def finalize(self):
        raise NotImplementedError()

    @property
    def last_segment(self):
        """Return the last segment of readable text since last time this property was accessed."""
        text = self.text
        if text and text[-1] != REPLACEMENT_CHAR:
            segment = text[self.offset :]
            self.offset = len(text)
            return segment
        return ""


class NaiveStreamingDetokenizer(StreamingDetokenizer):
    """NaiveStreamingDetokenizer relies on the underlying tokenizer
    implementation and should work with every tokenizer.

    Its complexity is O(T^2) where T is the longest line since it will
    repeatedly detokenize the same tokens until a new line is generated.
    """

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self._tokenizer.decode([0])
        self.reset()

    def reset(self):
        self.offset = 0
        self._tokens = []
        self._text = ""
        self._current_tokens = []
        self._current_text = ""

    def add_token(self, token):
        self._current_tokens.append(token)

    def finalize(self):
        self._tokens.extend(self._current_tokens)
        self._text += self._tokenizer.decode(self._current_tokens)
        self._current_tokens = []
        self._current_text = ""

    @property
    def text(self):
        if self._current_tokens:
            self._current_text = self._tokenizer.decode(self._current_tokens)
        if self._current_text and self._current_text[-1] == "\n":
            self._tokens.extend(self._current_tokens)
            self._text += self._current_text
            self._current_tokens.clear()
            self._current_text = ""
        return self._text + self._current_text

    @property
    def tokens(self):
        return self._tokens


class SPMStreamingDetokenizer(StreamingDetokenizer):
    """A streaming detokenizer for SPM models.

    It adds tokens to the text if the next token starts with the special SPM
    underscore which results in linear complexity.
    """

    def __init__(self, tokenizer, trim_space=True):
        self.trim_space = trim_space

        # Extract the tokens in a list from id to text
        self.tokenmap = [""] * (max(tokenizer.vocab.values()) + 1)
        for value, tokenid in tokenizer.vocab.items():
            self.tokenmap[tokenid] = value

        # Replace bytes with their value
        for i in range(len(self.tokenmap)):
            if self.tokenmap[i].startswith("<0x"):
                self.tokenmap[i] = chr(int(self.tokenmap[i][3:5], 16))

        self.reset()

    def reset(self):
        self.offset = 0
        self._unflushed = ""
        self.text = ""
        self.tokens = []

    def add_token(self, token):
        v = self.tokenmap[token]
        if v[0] == "\u2581":
            if self.text or not self.trim_space:
                self.text += self._unflushed.replace("\u2581", " ")
            else:
                self.text = _remove_space(self._unflushed.replace("\u2581", " "))
            self._unflushed = v
        else:
            self._unflushed += v

    def finalize(self):
        if self.text or not self.trim_space:
            self.text += self._unflushed.replace("\u2581", " ")
        else:
            self.text = _remove_space(self._unflushed.replace("\u2581", " "))
        self._unflushed = ""


class BPEStreamingDetokenizer(StreamingDetokenizer):
    """A streaming detokenizer for OpenAI style BPE models.

    It adds tokens to the text if the next token starts with a space similar to
    the SPM detokenizer.
    """

    _byte_decoder = None

    def __init__(self, tokenizer, trim_space=False):
        self.trim_space = trim_space

        # Extract the tokens in a list from id to text
        self.tokenmap = [None] * len(tokenizer.vocab)
        for value, tokenid in tokenizer.vocab.items():
            self.tokenmap[tokenid] = value

        self.reset()

        # Make the BPE byte decoder from
        # https://github.com/openai/gpt-2/blob/master/src/encoder.py
        self.make_byte_decoder()

    def reset(self):
        self.offset = 0
        self._unflushed = ""
        self.text = ""
        self.tokens = []

    def add_token(self, token):
        v = self.tokenmap[token]
        # if the token starts with space
        if self._byte_decoder[v[0]] == 32:
            current_text = bytearray(
                self._byte_decoder[c] for c in self._unflushed
            ).decode("utf-8")
            if self.text or not self.trim_space:
                self.text += current_text
            else:
                self.text += _remove_space(current_text)
            self._unflushed = v
        else:
            self._unflushed += v

    def finalize(self):
        current_text = bytearray(self._byte_decoder[c] for c in self._unflushed).decode(
            "utf-8"
        )
        if self.text or not self.trim_space:
            self.text += current_text
        else:
            self.text += _remove_space(current_text)
        self._unflushed = ""

    @classmethod
    def make_byte_decoder(cls):
        """See https://github.com/openai/gpt-2/blob/master/src/encoder.py for the rationale."""
        if cls._byte_decoder is not None:
            return

        char_to_bytes = {}
        limits = [
            0,
            ord("!"),
            ord("~") + 1,
            ord("¡"),
            ord("¬") + 1,
            ord("®"),
            ord("ÿ") + 1,
        ]
        n = 0
        for i, (start, stop) in enumerate(zip(limits, limits[1:])):
            if i % 2 == 0:
                for b in range(start, stop):
                    char_to_bytes[chr(2**8 + n)] = b
                    n += 1
            else:
                for b in range(start, stop):
                    char_to_bytes[chr(b)] = b
        cls._byte_decoder = char_to_bytes


class TokenizerWrapper:
    """A wrapper that combines a tokenizer and a detokenizer with transformers-compatible API.

    Provides transformers-compatible methods like encode() with return_tensors support
    while using the tokenizers library under the hood.
    """

    def __init__(self, tokenizer, detokenizer_class=NaiveStreamingDetokenizer):
        self._tokenizer = tokenizer
        self._detokenizer = detokenizer_class(tokenizer)

    def __getattr__(self, attr):
        if attr == "detokenizer":
            return self._detokenizer
        else:
            return getattr(self._tokenizer, attr)

    def _convert_to_tensors(self, token_ids, return_tensors=None):
        """Convert token IDs to the requested tensor format."""
        if return_tensors == "mlx":
            if isinstance(token_ids, list):
                return mx.array(token_ids)
            elif isinstance(token_ids[0], list):  # batch
                return mx.array(token_ids)
            return token_ids
        elif return_tensors is None:
            return token_ids
        else:
            raise ValueError(f"return_tensors='{return_tensors}' not supported. Use 'mlx' or None.")

    def encode(self, text, return_tensors=None, add_special_tokens=True, **kwargs):
        """Encode text with transformers-compatible API."""
        token_ids = self._tokenizer.encode(text, add_special_tokens=add_special_tokens).ids
        # Always add batch dimension for single text
        if return_tensors == "mlx":
            return mx.array([token_ids])  # Shape: [1, seq_len]
        elif return_tensors is None:
            return token_ids  # Keep as list for compatibility
        else:
            raise ValueError(f"return_tensors='{return_tensors}' not supported. Use 'mlx' or None.")

    def batch_encode_plus(self, texts, return_tensors=None, padding=False, truncation=False, 
                         max_length=None, add_special_tokens=True, **kwargs):
        """Batch encode texts with transformers-compatible API."""
        if isinstance(texts, str):
            texts = [texts]
        
        # Encode all texts
        all_token_ids = []
        for text in texts:
            encoding = self._tokenizer.encode(text, add_special_tokens=add_special_tokens)
            token_ids = encoding.ids
            
            # Apply truncation if specified
            if truncation and max_length and len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
            
            all_token_ids.append(token_ids)
        
        # Apply padding if specified
        if padding:
            if max_length is None:
                max_length = max(len(ids) for ids in all_token_ids)
            
            pad_token_id = (self._tokenizer.token_to_id('[PAD]') or 
                            self._tokenizer.token_to_id('<pad>') or 
                            self._tokenizer.token_to_id('<|endoftext|>') or 0)
            for i, token_ids in enumerate(all_token_ids):
                if len(token_ids) < max_length:
                    all_token_ids[i] = token_ids + [pad_token_id] * (max_length - len(token_ids))
        
        # Create attention masks
        attention_masks = []
        for token_ids in all_token_ids:
            mask = [1] * len(token_ids)
            attention_masks.append(mask)
        
        result = {
            'input_ids': self._convert_to_tensors(all_token_ids, return_tensors),
            'attention_mask': self._convert_to_tensors(attention_masks, return_tensors)
        }
        
        return result

    def decode(self, token_ids, skip_special_tokens=True, **kwargs):
        """Decode token IDs to text."""
        if hasattr(token_ids, 'tolist'):  # MLX array or similar
            token_ids = token_ids.tolist()
        return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def batch_decode(self, sequences, skip_special_tokens=True, **kwargs):
        """Batch decode sequences of token IDs."""
        if hasattr(sequences, 'tolist'):  # MLX array or similar
            sequences = sequences.tolist()
        
        results = []
        for seq in sequences:
            results.append(self._tokenizer.decode(seq, skip_special_tokens=skip_special_tokens))
        return results

    @property
    def pad_token_id(self):
        """Get the pad token ID."""
        return self._tokenizer.token_to_id('[PAD]') or self._tokenizer.token_to_id('<pad>') or 0

    @property
    def mask_token_id(self):
        """Get the mask token ID."""
        return self._tokenizer.token_to_id('[MASK]') or self._tokenizer.token_to_id('<mask>')

    @property
    def eos_token_id(self):
        """Get the end-of-sequence token ID."""
        return self._tokenizer.token_to_id('</s>') or self._tokenizer.token_to_id('<eos>')

    @property
    def vocab_size(self):
        """Get the vocabulary size."""
        return self._tokenizer.get_vocab_size()


def _match(a, b):
    if type(a) != type(b):
        return False
    if isinstance(a, dict):
        return len(a) == len(b) and all(k in b and _match(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_match(ai, bi) for ai, bi in zip(a, b))

    return a == b


def _is_spm_decoder(decoder):
    _target_description = {
        "type": "Sequence",
        "decoders": [
            {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
            {"type": "ByteFallback"},
            {"type": "Fuse"},
            {"type": "Strip", "content": " ", "start": 1, "stop": 0},
        ],
    }
    return _match(_target_description, decoder)


def _is_spm_decoder_no_space(decoder):
    _target_description = {
        "type": "Sequence",
        "decoders": [
            {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
            {"type": "ByteFallback"},
            {"type": "Fuse"},
        ],
    }
    return _match(_target_description, decoder)


def _is_bpe_decoder(decoder):
    _target_description = {
        "type": "ByteLevel",
        "add_prefix_space": False,
        "trim_offsets": False,
        "use_regex": False,
    }

    return _match(_target_description, decoder)


def load_tokenizer(tokenizer_path, tokenizer_config_extra={}):
    """Load a tokenizer from a local file path and try to infer the type of streaming
    detokenizer to use.

    Args:
        tokenizer_path: Path to the tokenizer.json file or directory containing it
        tokenizer_config_extra: Additional config parameters (kept for compatibility)
    """
    detokenizer_class = NaiveStreamingDetokenizer

    # Handle both direct file path and directory path
    if str(tokenizer_path).endswith('.json'):
        tokenizer_file = tokenizer_path
    else:
        tokenizer_file = tokenizer_path / "tokenizer.json"
    
    # Load tokenizer content to infer detokenizer type
    with open(tokenizer_file, "r") as fid:
        tokenizer_content = json.load(fid)
    
    if "decoder" in tokenizer_content:
        if _is_spm_decoder(tokenizer_content["decoder"]):
            detokenizer_class = SPMStreamingDetokenizer
        elif _is_spm_decoder_no_space(tokenizer_content["decoder"]):
            detokenizer_class = partial(SPMStreamingDetokenizer, trim_space=False)
        elif _is_bpe_decoder(tokenizer_content["decoder"]):
            detokenizer_class = BPEStreamingDetokenizer

    return TokenizerWrapper(
        Tokenizer.from_file(str(tokenizer_file)),
        detokenizer_class,
    )
