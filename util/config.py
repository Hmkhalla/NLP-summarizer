train_data_path = 	"data/chunked/train/train_*"
valid_data_path = 	"data/chunked/valid/valid_*"
test_data_path = 	"data/chunked/test/test_*"
vocab_path = 		"data/vocab"


# Hyperparameters
attn_dim = 1
hidden_dim = 100
emb_dim = 50
batch_size = 200
max_enc_steps = 25		#99% of the articles are within length 55
max_dec_steps = 8		#99% of the titles are within length 15
beam_size = 4
min_dec_steps= 3
vocab_size = 25000
datasize = 500000
lr = 0.001
rand_unif_init_mag = 0.02
trunc_norm_init_std = 1e-4

eps = 1e-12
max_epochs = 5


save_model_path = "data/saved_models"

intra_encoder = True
intra_decoder = True
