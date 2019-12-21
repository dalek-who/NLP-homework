import torch
import os
from pathlib import Path

from BaseAPI import BaseAPI
from models.ALBERT.model.modeling_albert import BertForSequenceClassification
from models.ALBERT.model.modeling_albert import BertConfig
from models.ALBERT.preprocessor import BertProcessor, InputExample

class ALBERT_API(BaseAPI):
    def __init__(self):
        super(ALBERT_API, self).__init__()
        self.path = Path(__file__)
        checkpoints_dir = os.path.dirname(os.path.abspath(__file__)) + "/checkpoints/"
        vocab_path = os.path.dirname(os.path.abspath(__file__)) + "/vocab.txt"
        bert_config = BertConfig.from_pretrained(checkpoints_dir + "config.json", share_type="all", num_labels=3)
        self.processor = BertProcessor(vocab_path=vocab_path, do_lower_case=False)
        self.model:torch.nn.Module = BertForSequenceClassification.from_pretrained(checkpoints_dir, config=bert_config)

    def run_example(self, text: str):
        preprocesspr = self.processor
        demo_examples = [InputExample(guid="demo-0", seq_id="x1", text_a=text, text_b=None, label=0)]
        demo_features = preprocesspr.create_features(examples=demo_examples, max_seq_len=511)
        all_seq_id = [f.seq_id for f in demo_features]
        all_input_ids = torch.tensor([f.input_ids for f in demo_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in demo_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in demo_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in demo_features], dtype=torch.long)
        self.model.eval()
        preds = []
        loss, logits = self.model(input_ids=all_input_ids, attention_mask=all_input_mask, labels=all_label_ids, token_type_ids=all_segment_ids)
        preds.append(logits.cpu().detach())
        preds = torch.cat(preds, dim=0).cpu().detach()
        preds_label = torch.argmax(preds, dim=1)
        return preds_label.item()


if __name__=="__main__":
    text1 = "热烈庆祝澳门回归二十周年\n习近平主席祝贺澳门回归二十周年,咱们老百姓新今天真高兴"
    text2 = "匪徒态度恶劣，藐视警察，还有王法吗？"
    model = ALBERT_API()
    result = model.run_example(text=text1)


