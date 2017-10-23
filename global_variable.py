# coding:utf-8

config = {
    "n_input": 200,
    "embedding_len": 200,
    "n_hidden_units": 128,
    "n_hidden_layers": 5,

    "max_query_len": 667,
    "max_response_len": 304,

    "batch_size": 128,
    "max_training_iters": 10000000,
    "checkout_iters": 1000000,

    "learning_rate": 0.001,
    "keep_prob": 0.6,

    'AE_completed': 100000,
}

class GlobalVariable(object):
    # 语料库的所有记录
    corpus_sets = []
    # 语料库的总记录数量
    corpus_sets_num = 0
    # 词向量字典
    embedding_dict = None
    # 打乱次序后的下标所构成的列表
    shuffle_index = []

    # 所遍历的记录的下标
    index_in_epoch = 0
    # epoch_size = 0
    # # 整个语料库的记录数的大小
    # num_examples = 0
    # 已经进行了多少次的epoch循环，即已经进行了多少次的的整个语料库
    epochs_completed = 1
    # eopch = 0

    # 测试相关
    test_corpus = []

    sentence2Words = None
    wordToVector = None

    model_save_path = './model/MTEnDecoder.ckpt'
    MTEnDecoder_save_path = './model/MTEnDecoder.ckpt'
    MTSemanticLogicED_save_path = './model/MTSemanticLogicED.ckpt'

    ubuntu_dialogs_path = '/tmp/ubuntu_dialogs/dialogs'
    corpus_file_path = './corpus/QR_records.json'
    corpus_file_path_mini = './corpus/QR_records_mini.json'

    placeholder = 'placeholder'
    ph_embedding = None

    bin_file_path = "./embedding/vectors.bin"

class ValidOrTestParm(object):
    # 语料库的所有记录
    corpus_sets = []
    # 语料库的总记录数量
    corpus_sets_num = None
    # 词向量字典
    embedding_dict = None
    # 所遍历的记录的下标
    index_in_epoch = 0
    # 已经进行了多少次的epoch循环，即已经进行了多少次的的整个语料库
    epochs_completed = 1
