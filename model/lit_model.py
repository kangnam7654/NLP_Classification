import torch
from utils.common.project_paths import GetPaths
from pytorch_lightning import LightningModule
from transformers import AutoModelForSeq2SeqLM, BartForConditionalGeneration, BartTokenizerFast


class LitModel(LightningModule):
    def __init__(self, mode='train'):
        super().__init__()
        self.bart_config = self.__bart_configs()
        self.new_encoder = None
        self.__build_model()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.mode = mode

    def forward(self, x, y=None):
        if self.mode == 'train':
            out = self.model(input_ids=x, labels=y)
        else:
            out = self.model(input_ids=x)
        return out

    def training_step(self, batch, batch_idx):
        train_loss, train_acc = self.__share_step(batch)
        # train_acc.requires_grad = True
        results = {'loss': train_loss, 'acc': train_acc}
        return results

    def validation_step(self, batch, batch_idx):
        valid_loss, valid_acc = self.__share_step(batch)
        results = {'loss': valid_loss, 'acc': valid_acc}
        return results

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        out = torch.softmax(self(batch), dim=1)
        pred = torch.argmax(out, dim=1)
        return pred

    def training_epoch_end(self, outputs):
        self.__share_epoch_end(outputs, 'train')

    def validation_epoch_end(self, outputs):
        self.__share_epoch_end(outputs, 'valid')

    def configure_optimizers(self):
        opt = torch.optim.SGD(
            self.parameters(),
            lr=0.001,
            weight_decay=0.0001,
            momentum=0.9,
            nesterov=True,
        )
        # sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=opt,)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=opt,
                                                         mode='min',
                                                         factor=0.1,
                                                         patience=2,
                                                         min_lr=1e-8,
                                                         verbose=True)
        monitor = 'valid_loss'
        returns = {'optimizer': opt, 'lr_scheduler': sch, 'monitor': monitor}
        return returns

    def __share_step(self, batch):
        data, label = batch
        out = self(data, label)
        loss = out.loss
        acc = self.compute_accuracy(out, label).unsqueeze(dim=0)
        return loss, acc

    def __share_epoch_end(self, outputs, mode):
        all_loss = []
        all_acc = []
        for out in outputs:
            loss, acc = out['loss'], out['acc']
            all_loss.append(loss.unsqueeze(0))
            all_acc.append(acc.unsqueeze(0))
        avg_loss = torch.mean(torch.cat(all_loss))
        avg_acc = torch.mean(torch.cat(all_acc))
        self.log_dict({f'{mode}_loss': avg_loss, f'{mode}_acc': avg_acc})

    def __bart_configs(self):
        bart_config = AutoModelForSeq2SeqLM.from_pretrained('facebook/bart-base').config
        bart_config.vocab_size = 64000  # default : 50265
        return bart_config

    def __build_model(self):
        self.model = BartForConditionalGeneration(self.bart_config)
        self.model.on_gpu = True
        self.tokenizer = BartTokenizerFast(tokenizer_file=GetPaths.get_project_root('custom_tokenizer', 'vocab', 'vocab.json'))

    def apply_ckpt(self, checkpoint_path):
        ckpt = torch.load(checkpoint_path)
        if 'state_dict' in ckpt.keys():
            state_dict = {}
            for k, v in ckpt['state_dict'].items():
                k = k[6:]
                state_dict[k] = v
            self.model.load_state_dict(state_dict)
        else:
            self.model.load_state_dict(ckpt)
        print(f'모델을 성공적으로 불러왔습니다.')

    @staticmethod
    def compute_accuracy(out, labels):  # for classification
        max_indices = torch.argmax(torch.softmax(out.logits, dim=2), dim=2)
        acc = (max_indices == labels).to(torch.float).mean() * 100
        return acc


if __name__ == '__main__':
    model = LitModel()