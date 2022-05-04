import torch


def apply_ckpt(model, checkpoint_path):
    ckpt = torch.load(checkpoint_path)
    if 'state_dict' in ckpt.keys():
        state_dict = {}
        for k, v in ckpt['state_dict'].items():
            k = k.replace('model.', '')
            state_dict[k] = v
        model.load_state_dict(state_dict)
    else:
        model.load_state_dict(ckpt)

    print(f'모델을 성공적으로 불러왔습니다.')
    return model


def apply_device(model, device):
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 10:
            print("Multi-Device")
            model = torch.nn.DataParallel(model).to(device)
        else:
            model = model.to(device)
    else:
        model = model.to(device)
    return model


def load_roberta(config, device):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    model_name = config['MODEL']['NAME']
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=config['MODEL']['N_CLASSES'])

    checkpoint_path = config['MODEL']['CHECKPOINT']
    if checkpoint_path is not None:
        model = apply_ckpt(model, checkpoint_path)
    model = apply_device(model, device)
    return model, tokenizer


if __name__ == '__main__':
    pass