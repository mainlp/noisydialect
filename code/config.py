class Config:
    __slots__ = 'name_train', 'name_dev', 'name_test', \
                'prepare_input_traindev', \
                'orig_file_traindev', 'orig_file_test', 'max_sents_traindev', \
                'max_sents_test', 'dev_ratio', \
                'orig_dir_train', 'orig_dir_dev', \
                'T', 'tokenizer_name', \
                'noise_type', 'noise_lvl_min', 'noise_lvl_max', \
                'data_parent_dir', 'bert_name', \
                'classifier_dropout', 'n_epochs', 'batch_size'

    ints = ['max_sents_traindev', 'max_sents_test', 'T', 'n_epochs',
            'batch_size']
    floats = ['dev_ratio', 'noise_lvl_min' 'noise_lvl_max',
              'classifier_dropout']
    bools = ['prepare_input_traindev']

    def __init__(self,
                 name_train=None,
                 name_dev=None,
                 name_test=None,
                 # Loading data:
                 orig_file_traindev=None,
                 orig_file_test=None,
                 max_sents_traindev=-1,  # -1: no max limit
                 max_sents_test=-1,  # -1: no max limit
                 dev_ratio=0.1,
                 orig_dir_train=None,
                 orig_dir_dev=None,
                 # If the input matrices still need to be prepared:
                 prepare_input_traindev=False,
                 T=60,
                 tokenizer_name="dbmdz/bert-base-german-cased",
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
                 ):
        self.prepare_input_traindev = prepare_input_traindev
        self.name_train = name_train
        self.name_dev = name_dev
        self.name_test = name_test
        self.orig_file_traindev = orig_file_traindev
        self.orig_file_test = orig_file_test
        self.max_sents_traindev = max_sents_traindev
        self.max_sents_test = max_sents_test
        self.dev_ratio = dev_ratio
        self.orig_dir_train = orig_dir_train
        self.orig_dir_dev = orig_dir_dev
        self.T = T
        self.tokenizer_name = tokenizer_name
        self.noise_type = noise_type
        self.noise_lvl_min = noise_lvl_min
        self.noise_lvl_max = noise_lvl_max
        self.data_parent_dir = data_parent_dir
        self.bert_name = bert_name
        self.classifier_dropout = classifier_dropout
        self.n_epochs = n_epochs
        self.batch_size = batch_size

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
