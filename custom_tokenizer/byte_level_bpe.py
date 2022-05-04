from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from transformers import BartTokenizerFast, BartTokenizer
from utils.common.project_paths import GetPaths
from glob import glob


def main():
    # Initialize a tokenizer
    tokenizer = Tokenizer(models.BPE())

    # Customize pre-tokenization and decoding
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

    # Special Tokens
    special_token = ['<s>', '<pad>', '</s>', '<unk>', '<mask>']

    # And then train
    trainer = trainers.BpeTrainer(
        vocab_size=64000,
        min_frequency=5,
        limit_alphabet=6000,
        special_tokens=special_token
    )
    corp_path = glob(GetPaths.get_data_folder('*.txt'), recursive=True)
    tokenizer.train(corp_path, trainer=trainer)

    # And Save it
    tokenizer.save("./vocab/vocab.json", pretty=True)


def test():
    tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-base')
    tokenizer2 = BartTokenizerFast(tokenizer_file='./vocab/vocab.json')
    encode = tokenizer.encode('').ids
    decode = tokenizer.decode(encode)
    print(f'encode: {encode}, decode: {decode}')


if __name__ == '__main__':
    main()
    # test()