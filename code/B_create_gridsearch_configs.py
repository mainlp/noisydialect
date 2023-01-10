from config import Config
import sys


def hyperparams():
    for conf_file in ("B_hdt-full_hdt-noah_gbert_orig.cfg",
                      "B_padt-full_padt-egy_arabert_orig.cfg",
                      "B_tdt-full_tdt-sav_finbert_orig.cfg",
                      "B_hdt-full_hdt-noah_xlmr_orig.cfg",
                      "B_padt-full_padt-egy_xlmr_orig.cfg",
                      "B_tdt-full_tdt-sav_xlmr_orig.cfg"):
        for lr in (2e-5, 3e-5):
            for batch_size in (16, 32):
                # Epochs up to n_epochs are checked anyway.
                conf = Config("dummy")
                conf.load("../configs/" + conf_file)
                conf.config_name = "B_hyperparams_" + conf.config_name[2:] \
                                   + "_" + str(lr) + "_" + str(batch_size)
                conf.learning_rate = lr
                conf.batch_size = batch_size
                conf.n_epochs = 3
                conf.prepare_input_train = False
                conf.prepare_input_dev = False
                conf.prepare_input_test = False
                conf.save("../configs/" + conf.config_name + ".cfg")
                print(conf.config_name)


def models():
    lr = 2e-5
    batch_size = 32
    epochs = 2
    all_configs = []
    for conf_file in (
            "C_hdt-full_hdt-noah_gbert_orig.cfg",
            "C_alpino-full_alpino-noah_bertje_orig.cfg",
            "C_gsd-full_gsd-rpic_camembert_orig.cfg",
            "C_ancoraspa-full_ancoraspa-rpic_beto_orig.cfg",
            "C_nob-full_nob-west_norbert_orig.cfg",
            "C_nno-full_nno-west_norbert_orig.cfg",
            "C_padt-full_padt-egy_arabert_orig.cfg",
            "C_padt-translit-full_padt-translit_bertu_orig.cfg",
            "C_tdt-full_tdt-sav_finbert_orig.cfg",
    ):
        all_configs.append(conf_file[:-4])
        for name_short, name_long, model_type in (
                ("mbert", "bert-base-multilingual-cased", "BertForMaskedLM"),
                ("xlmr", "xlm-roberta-base", "XLMRobertaForMaskedLM")):
            conf = Config("dummy")
            conf.load("../configs/" + conf_file)
            pfx, old_model, sfx = conf.config_name.rsplit("_", 2)
            conf.config_name = pfx + "_" + name_short + "_" + sfx
            conf.learning_rate = lr
            conf.batch_size = batch_size
            conf.n_epochs = epochs
            conf.name_train = conf.name_train.replace(old_model, name_short)
            try:
                conf.name_dev = conf.name_dev.replace(old_model, name_short)
            except AttributeError:
                conf.name_dev = [dev.replace(old_model, name_short)
                                 for dev in conf.name_dev]
            conf.bert_name = name_long
            conf.tokenizer_name = name_long
            conf.plm_type = model_type
            conf.prepare_input_train = False
            conf.prepare_input_dev = False
            conf.prepare_input_test = False
            conf.save("../configs/" + conf.config_name + ".cfg")
            print(conf.config_name)
            all_configs.append(conf.config_name)
    return all_configs


def noise(configs):
    for conf_name in configs:
        for noise_lvl in (0.15, 0.35, 0.55, 0.75, 0.95):
            conf = Config("dummy")
            conf.load("../configs/" + conf_name + ".cfg")
            pfx, _ = conf.config_name.rsplit("_", 1)
            noise_name = "_rand" + str(int(noise_lvl * 100))
            conf.config_name = pfx + noise_name
            conf.name_train = conf.name_train.replace("_orig", noise_name)
            conf.noise_lvl = noise_lvl
            conf.noise_type = "add_random_noise"
            conf.prepare_input_train = True
            conf.reinit_train_each_seed = True
            conf.save("../configs/" + conf.config_name + ".cfg")
            print(conf.config_name)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Expected one argument: "
              "'--hyperparams' or '--noise' or '--noiseonly'")
        sys.exit(1)
    if sys.argv[1].endswith("hyperparams"):
        hyperparams()
    elif sys.argv[1].endswith("noise"):
        configs = models()
        noise(configs)
    else:
        configs = ("C_hdt-full_hdt-noah_bertje_orig",
                   "C_alpino-full_alpino-noah_gbert_orig",
                   "C_gsd-full_gsd-rpic_beto_orig",
                   "C_ancoraspa-full_ancoraspa-rpic_camembert_orig",
                   "C_tdt-full_tdt-sav_estbert_orig",
                   "C_tdt-full_tdt-sav_bert_orig",
                   "C_nno-full_nno-west_bert_orig",
                   "C_nno-full_nno-west_finbert_orig",
                   "C_nno-full_nno-west_arabert_orig",)
        noise(configs)
