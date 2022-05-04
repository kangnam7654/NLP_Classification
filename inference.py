from model.lit_model import LitModel
import torch


def main():
    model = LitModel(mode='inference')
    model.apply_ckpt('./ckpt/model.ckpt')
    tokenizer = model.tokenizer

    # kor = input('원문 입력: ')
    kor = '안녕하세요'

    token = torch.tensor(tokenizer(kor).input_ids).unsqueeze(0)
    out = torch.argmax(torch.softmax(model(token).logits, dim=2), dim=2)
    translated = tokenizer.decode(list(out)[0])

    print(translated)


if __name__ == '__main__':
    main()