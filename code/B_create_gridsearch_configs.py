from config import Config

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
