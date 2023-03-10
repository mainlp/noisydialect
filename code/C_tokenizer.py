import re
import unidecode
from transformers import BertTokenizer, WordpieceTokenizer
from collections import OrderedDict


class SCATokenizer(BertTokenizer):

    def __init__(self, pretrained_base_name):
        tokenizer = BertTokenizer.from_pretrained(pretrained_base_name)
        self.__dict__.update(tokenizer.__dict__)
        self.name_or_path = tokenizer.name_or_path.replace("/", "_") + "_sca"
        self.special_tokens = self.additional_special_tokens.copy()
        for attr in self.SPECIAL_TOKENS_ATTRIBUTES:
            # [UNK], [SEP], etc.
            if attr != "additional_special_tokens":
                attr_value = getattr(self, "_" + attr)
                if attr_value:
                    self.special_tokens.append(attr_value)
        self.vocab = dict()
        self.max_siblings = 1
        for tok, idx in tokenizer.vocab.items():
            if tok in self.special_tokens:
                self.vocab[tok] = [idx]
                continue
            if tok.startswith("unused"):
                # 'unusedXX' = additional group of uninitiated tokens
                try:
                    int(tok[len("unused"):])
                    self.vocab[tok] = [idx]
                    continue
                except ValueError:
                    pass
            # TODO let the user specify which orthography is to be used
            tok_sca = self.deu2sca(tok)
            siblings = self.vocab.get(tok_sca, [])
            if len(siblings) >= self.max_siblings:
                self.max_siblings = len(siblings) + 1
            siblings.append(idx)
            self.vocab[tok_sca] = siblings
        ids_to_tokens_unsorted = {}
        for tok, idxs in self.vocab.items():
            for idx in idxs:
                ids_to_tokens_unsorted[idx] = tok
        self.ids_to_tokens = OrderedDict()
        for i in range(len(tokenizer.vocab)):
            self.ids_to_tokens[i] = ids_to_tokens_unsorted[i]
        self.wordpiece_tokenizer = WordpieceTokenizer(
            vocab=self.vocab, unk_token=self.unk_token)
        del tokenizer

    # Overrides PreTrainedTokenizer method
    def prepare_for_tokenization(self, text, is_split_into_words=False,
                                 **kwargs):
        if is_split_into_words:
            words_sca = []
            for word in text:
                words_sca.append(self.deu2sca(word))
            return words_sca
        return self.deu2sca(text), kwargs

    @staticmethod
    def deu2sca(text, reduce_twins=True):
        # Based on
        # http://lingulist.de/documents/list-2012-sca.pdf

        # Table from
        # https://github.com/lingpy/lingpy/blob/v2.6.9/src/lingpy/data/models/sca/converter
        # (GPL 3.0)
        # A : a, ᴀ, ã, ɑ, á, á, à, ā, ǎ, â, ä, ă, ă, ạ, а, å, ạ
        # C : t͡s, t͜s, d͡z, d͜z, ʦ, ʣ, t͡ɕ, t͜ɕ, d͡ʑ, d͜ʑ, ʨ, ʥ, t͡ʃ, t͜ʃ,
        #     d͡ʒ, d͜ʒ, ʧ, ʤ, c, ɟ, t͡ʂ, t͜ʂ, d͡ʐ, d͜ʐ, č, t͡θ, t͜θ, ʄ, ǰ,
        #     ĵ, Ɉ, ʈʂ, ɖʐ, ʈʂʰ, tɕ, tɕʰ, dʑ, ts, dz, tsʰ
        # B : ɸ, β, f, p͡f, p͜f, ƀ, p͡f, b͡v, pf, bv, v, ʙ, ḇ
        # E : ɛ, æ, ɜ, ɐ, ʌ, e, ᴇ, ə, ɘ, ɤ, è, é, ē, ě, ê, ɚ, ǝ, ẽ, ĕ, ḛ, ε, е,
        #     ę, ḛ, ȇ, ë, ε
        # D : θ, ð, ŧ, þ, đ, Ɵ
        # G : x, ɣ, χ
        # I : i, ɪ, ɨ, ɿ, ʅ, ɯ, ĩ, í, ǐ, ì, î, ī, ı, ĭ, ḭ, ɩ, ï, ị, ๅ, ḭ
        # H : ʔ, ħ, ʕ, h, ɦ, ḥ, Ɂ, ʡ, 'ʷ
        # K : k, g, q, ɢ, ɡ, ḳ, ǥ, ǵ, ḡ, ɠ, ʛ
        # J : j, ɥ, ɰ
        # M : m, ɱ, ʍ, ṃ
        # L : l, ȴ, l, ɭ, ʎ, ʟ, ɬ, ɮ, ł, ɫ, ḷ, ļ, ᶅ
        # O : Œ, ɒ
        # N : n, ȵ, ɳ, ŋ, ɴ, ň, ń, ɲ, ñ, ṇ, ῃ, ņ, ṋ, ᶇ
        # P : p, b, ɓ, ᵐb, ᵐp, р
        # S : ʆ, s, z, ʃ, ∫, ʒ, ʂ, ʐ, ç, ʝ, š, ž, ɕ, ɧ, ʑ, ś, ṣ,
        #     ß, ŝ, ż, ẓ, ᶊ, ᶎ
        # R : ɹ, ɻ, ʀ, ɾ, r, ʁ, ɽ, ɐ̯, ɺ, ṛ, ᵲ, ř, ȓ, ṛ́, ṙ, ᶉ
        # U : œ, ɞ, ɔ, ø, ɵ, o, õ, ó, ò, ō, ɶ, ô, ɷ, ǒ, ö, ŏ, ʮ, ọ, ȯ, ố, ǫ, ṍ
        # T : t, d, ȶ, ȡ, ɗ, ʈ, ɖ, ţ, т, ṱ, ṭ, ḍ, ḏ, ᶁ, ƫ
        # W : w, ʋ, ⱱ, ṿ, υ, ṽ
        # Y : y, ỹ, ỹ, ṳ, ṵ, ʏ, ʉ, u, ᴜ, ʊ, ú, ù, ũ, ü, ŭ, ǔ, ụ, ū, ỳ,
        #     û, û, ý, ў, ȗ, ṹ, ṳ, ŷ, ʯ

        text = text.lower()
        text = unidecode.unidecode(text)
        # Some replacements need to be applied before others

        text = text.replace("qu", "kb")
        # A - unrounded [open] vowels
        text = text.replace("ah", "a")
        text = text.replace("a", "a")
        # I - unrounded close vowels
        text = text.replace("ieh", "i")
        text = text.replace("ih", "i")
        text = text.replace("ie", "i")
        text = text.replace("i", "i")
        # O - rounded back vowels -- n/a
        # U - rounded mid vowels
        text = text.replace("oeh", "u")
        text = text.replace("oe", "u")
    #     text = text.replace("öh", "u")
    #     text = text.replace("ö", "u")
        text = text.replace("oh", "u")
        text = text.replace("o", "u")
        # Y - rounded [close] vowels
        text = text.replace("ueh", "y")
        text = text.replace("ue", "y")
    #     text = text.replace("üh", "y")
    #     text = text.replace("ü", "y")
        text = text.replace("uh", "y")
        text = text.replace("u", "y")
        text = text.replace("y", "y")
        # E - unrounded mid vowels
        text = text.replace("aeh", "e")
        text = text.replace("ae", "e")
    #     text = text.replace("äh", "e")
    #     text = text.replace("ä", "e")
        text = text.replace("eh", "e")
        text = text.replace("e", "e")

        # Since pairs of voiced/voiceless consonants are in the same
        # sound classes, we don't need to worry about final devoicing.
        # B - labial fricatives
        text = text.replace("pf", "b")
        text = text.replace("ph", "b")
        text = text.replace("f", "b")
        text = text.replace("v", "b")
        text = text.replace("w", "b")
        # C - dental/alveolar affricates
        text = text.replace("tz", "c")
        text = text.replace("z", "c")
        text = text.replace("tsch", "c")
        text = text.replace("dsch", "c")
        text = text.replace("ts", "c")
        # D - dental fricatives -- n/a
        # S - sibilant fricatives
        text = text.replace("sch", "s")
        text = text.replace("chs", "ks")
        text = text.replace("s", "s")
        # Note that /ʃ/ and /z/, /s/ are in the same group,
        # so no need to worry about syllable-initial <st>, <sp>
        text = text.replace("ß", "s")
        # G - velar/uvular fricatives
        # technically, this should only include [x] but not [ç] (s)
        text = text.replace("ch", "g")
        # J - palatal approximant
        text = text.replace("j", "j")
        # N - (other) nasals
        text = text.replace("ng", "n")
        text = text.replace("n", "n")
        # K - velar/uvular plosives
        text = text.replace("ck", "k")
        text = text.replace("k", "k")
        text = text.replace("g", "k")
        # L - lateral approximants
        text = text.replace("l", "l")
        # M - labial nasal
        text = text.replace("m", "m")
        # P - labial plosives
        text = text.replace("p", "p")
        text = text.replace("b", "p")
        # R - trills, taps, flaps
        text = text.replace("rh", "r")
        text = text.replace("r", "r")
        # T - dental/alveolar plosives
        text = text.replace("th", "t")
        text = text.replace("t", "t")
        text = text.replace("d", "t")
        # W - labial approximants -- n/a
        # H - laryngeals
        text = text.replace("h", "h")

        text = text.replace("x", "ks")
        text = text.replace("c", "k")
        text = text.replace("q", "k")

        # This doesn't take into account:
        # - loanwords
        # - s+ch ("Mäus+chen")

        if reduce_twins:
            # Not SCA: reduce double/triple/etc sounds:
            text = re.sub(r"([a-z])\1+", r"\1", text)
        return text
