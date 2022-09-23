class Config:
    __slots__ = 'name_train', 'name_dev', 'name_test', \
                'prepare_input_traindev', 'prepare_input_test', \
                'orig_file_traindev', 'orig_file_test', \
                'tagset_path', \
                'encoding_traindev', 'encoding_test', \
                'max_sents_traindev', \
                'max_sents_test', 'dev_ratio', \
                'orig_dir_train', 'orig_dir_dev', \
                'T', 'subtoken_rep', 'tokenizer_name', \
                'use_sca_tokenizer', 'sca_sibling_weighting', \
                'noise_type', 'noise_lvl_min', 'noise_lvl_max', \
                'data_parent_dir', 'bert_name', \
                'classifier_dropout', 'n_epochs', 'batch_size', \
                'learning_rate', 'weight_decay', 'sanity_mod'

    ints = ['max_sents_traindev', 'max_sents_test', 'T', 'n_epochs',
            'batch_size', 'sanity_mod']
    floats = ['dev_ratio', 'noise_lvl_min', 'noise_lvl_max',
              'classifier_dropout', 'learning_rate', 'weight_decay']
    bools = ['prepare_input_traindev', 'prepare_input_test',
             'use_sca_tokenizer']

    def __init__(self,
                 name_train=None,
                 name_dev=None,
                 name_test=None,
                 # Loading data:
                 orig_file_traindev=None,
                 orig_file_test=None,
                 encoding_traindev="utf8",
                 encoding_test="utf8",
                 max_sents_traindev=-1,  # -1: no max limit
                 max_sents_test=-1,  # -1: no max limit
                 dev_ratio=0.1,
                 orig_dir_train=None,
                 orig_dir_dev=None,
                 tagset_path="tagset_stts.txt",
                 # If the input matrices still need to be prepared:
                 prepare_input_traindev=False,
                 prepare_input_test=False,
                 T=60,
                 subtoken_rep='last',  # 'first', 'last', 'all'
                 tokenizer_name="dbmdz/bert-base-german-cased",
                 # If use_sca_tokenizer == True, tokenizer_name is
                 # the name of the base tokenizer from which the
                 # SCATokenizer is built.
                 use_sca_tokenizer=False,
                 sca_sibling_weighting="relative",  # 'mean', 'relative'
                 noise_type=None,  # None -> no noise
                 noise_lvl_min=0.1,
                 noise_lvl_max=0.15,
                 # If the input matrices just need to be loaded:
                 data_parent_dir="../data/",
                 # Model:
                 bert_name="dbmdz/bert-base-german-cased",
                 # TODO: continued pretraining
                 classifier_dropout=0.1,
                 # Training:
                 n_epochs=2,
                 batch_size=32,
                 learning_rate=2e-5,
                 weight_decay=0.01,
                 sanity_mod=1000,
                 ):
        self.prepare_input_traindev = prepare_input_traindev
        self.prepare_input_test = prepare_input_test
        self.name_train = name_train
        self.name_dev = name_dev
        self.name_test = name_test
        self.orig_file_traindev = orig_file_traindev
        self.orig_file_test = orig_file_test
        self.encoding_traindev = encoding_traindev
        self.encoding_test = encoding_test
        self.max_sents_traindev = max_sents_traindev
        self.max_sents_test = max_sents_test
        self.dev_ratio = dev_ratio
        self.orig_dir_train = orig_dir_train
        self.orig_dir_dev = orig_dir_dev
        self.tagset_path = tagset_path
        self.T = T
        self.subtoken_rep = subtoken_rep
        self.tokenizer_name = tokenizer_name
        self.use_sca_tokenizer = use_sca_tokenizer
        self.sca_sibling_weighting = sca_sibling_weighting
        self.noise_type = noise_type
        self.noise_lvl_min = noise_lvl_min
        self.noise_lvl_max = noise_lvl_max
        self.data_parent_dir = data_parent_dir
        self.bert_name = bert_name
        self.classifier_dropout = classifier_dropout
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.sanity_mod = sanity_mod

    def save(self, path):
        with open(path, "w", encoding="utf8") as f:
            for attr in self.__slots__:
                f.write(f"{attr}\t{getattr(self, attr)}\n")

    def load(self, path):
        with open(path, "r", encoding="utf8") as f:
            for line in f:
                cells = line.strip().split("\t")
                try:
                    val = cells[1]
                    if val == "None":
                        val = None
                    elif cells[0] in self.ints:
                        val = int(val)
                    elif cells[0] in self.floats:
                        val = float(val)
                    elif cells[0] in self.bools:
                        val = cells[1] == "True"
                    setattr(self, cells[0], val)
                except AttributeError:
                    print(f"Key {cells[0]} is unknown (skipping)")

    def __str__(self):
        return "\n".join(
            f"{attr}: {getattr(self, attr)}" for attr in self.__slots__)
