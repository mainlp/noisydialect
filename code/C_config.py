class Config:
    __slots__ = 'config_name', 'name_train', 'name_dev', 'name_test', \
                'tagset_path', \
                'max_sents_train', 'max_sents_dev', 'max_sents_test', \
                'subset_selection', \
                'orig_dir_train', 'orig_dir_dev', 'T', \
                'orig_file_train', 'orig_file_dev', 'orig_file_test', \
                'prepare_input_train', \
                'prepare_input_dev', 'prepare_input_test', \
                'reinit_train_each_seed', \
                'reinit_dev_each_seed', 'reinit_test_each_seed', \
                'subtoken_rep', 'tokenizer_name', \
                'use_sca_tokenizer', 'sca_sibling_weighting', \
                'noise_type', 'noise_lvl', \
                'data_parent_dir', 'bert_name', 'plm_type',\
                'classifier_dropout', 'n_epochs', 'batch_size', \
                'learning_rate', 'sanity_mod', \
                'random_seeds'

    ints = ('max_sents_train', 'max_sents_dev', 'max_sents_test', 'T',
            'n_epochs', 'batch_size', 'sanity_mod')
    floats = ('noise_lvl', 'classifier_dropout', 'learning_rate')
    bools = ('prepare_input_train', 'prepare_input_dev', 'prepare_input_test',
             'reinit_train_each_seed', 'reinit_dev_each_seed',
             'reinit_test_each_seed', 'use_sca_tokenizer')
    lists_of_ints = ('random_seeds')

    def __init__(self,
                 config_name=None,
                 name_train=None,
                 name_dev=None,  # can be a comma-separated list
                 name_test=None,  # can be a comma-separated list
                 # Loading data:
                 orig_file_train=None,
                 orig_file_dev=None,
                 orig_file_test=None,  # can be a comma-separated list
                 max_sents_train=-1,  # -1: no max limit
                 max_sents_dev=-1,  # -1: no max limit
                 max_sents_test=-1,  # -1: no max limit
                 subset_selection=None,  # {first, last, rand}
                 orig_dir_train=None,
                 orig_dir_dev=None,
                 tagset_path=None,
                 # If the input matrices still need to be prepared:
                 prepare_input_train=False,
                 prepare_input_dev=False,
                 prepare_input_test=False,
                 # Same for the reinit... values.
                 reinit_train_each_seed=False,
                 reinit_dev_each_seed=False,
                 reinit_test_each_seed=False,
                 T=-1,
                 subtoken_rep=None,  # {first, last, all}
                 tokenizer_name=None,
                 # None -> no noise.
                 # {add_random_noise, add_custom_noise_general}
                 noise_type=None,
                 noise_lvl=0,
                 # If the input matrices just need to be loaded:
                 data_parent_dir=None,
                 # Model:
                 bert_name=None,
                 plm_type=None,
                 # TODO: continued pretraining
                 classifier_dropout=0.1,
                 # Training:
                 n_epochs=2,
                 batch_size=32,
                 learning_rate=2e-5,
                 sanity_mod=1000,
                 random_seeds=[12345, 23456, 34567, 45678, 56789]
                 ):
        self.config_name = config_name
        self.name_train = name_train
        self.name_dev = name_dev
        self.name_test = name_test
        self.orig_file_train = orig_file_train
        self.orig_file_dev = orig_file_dev
        self.orig_file_test = orig_file_test
        self.max_sents_train = max_sents_train
        self.max_sents_dev = max_sents_dev
        self.max_sents_test = max_sents_test
        self.subset_selection = subset_selection
        self.orig_dir_train = orig_dir_train
        self.orig_dir_dev = orig_dir_dev
        self.tagset_path = tagset_path
        self.T = T
        self.prepare_input_train = prepare_input_train
        self.prepare_input_dev = prepare_input_dev
        self.prepare_input_test = prepare_input_test
        self.reinit_train_each_seed = reinit_train_each_seed
        self.reinit_dev_each_seed = reinit_dev_each_seed
        self.reinit_test_each_seed = reinit_test_each_seed
        self.subtoken_rep = subtoken_rep
        self.tokenizer_name = tokenizer_name
        self.use_sca_tokenizer = use_sca_tokenizer
        self.sca_sibling_weighting = sca_sibling_weighting
        self.noise_type = noise_type
        self.noise_lvl = noise_lvl
        self.data_parent_dir = data_parent_dir
        self.bert_name = bert_name
        # plm_type options: BertForMaskedLM, BertForPreTraining,
        # RobertaForMaskedLM, XLMRobertaForMaskedLM
        self.plm_type = plm_type
        self.classifier_dropout = classifier_dropout
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.sanity_mod = sanity_mod
        self.random_seeds = random_seeds

    def save(self, path):
        with open(path, "w+", encoding="utf8") as f:
            for attr in self.__slots__:
                f.write(f"{attr}\t{getattr(self, attr)}\n")

    def load(self, path):
        with open(path, "r", encoding="utf8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("#"):
                    continue
                cells = line.split("\t")
                try:
                    key = cells[0]
                    try:
                        val = cells[1]
                    except IndexError:
                        val = ""
                    if val == "None":
                        val = None
                    elif key in self.ints:
                        val = int(val)
                    elif key in self.floats:
                        val = float(val)
                    elif key in self.bools:
                        val = cells[1] == "True"
                    elif key in self.lists_of_ints:
                        val = [int(entry.strip())
                               for entry in val.strip()[1:-1].split(',')]
                    setattr(self, key, val)
                except AttributeError:
                    print(f"Key {cells[0]} is unknown (skipping)")

    def compare(self, other):
        found_differences = False
        for attr in self.__slots__:
            if getattr(self, attr) != getattr(other, attr):
                found_differences = True
                print(f"Different values for {attr}!")
                print(getattr(self, attr))
                print(getattr(other, attr))
        if not found_differences:
            print("Identical configs")

    def __str__(self):
        return "\n".join(
            f"{attr}: {getattr(self, attr)}" for attr in self.__slots__)
