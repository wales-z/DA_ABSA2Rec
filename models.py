import numpy as np
import torch
import torch.nn as nn
import torchvision
# print(torch.__version__) # 1.10.0+cu102
# print(torch.version.cuda) # 10.2
# print(torch.backends.cudnn.version()) # 7605
# print(torch.cuda.get_device_name(0)) # GeForce RTX 2080 Ti

# 固定随机种子以保证可复现性
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TaggerConfig:
    def __init__(self):
        self.hidden_dropout_prob = 0.1
        self.hidden_size = 768
        self.n_rnn_layers = 1  # not used if tagger is non-RNN model
        self.bidirectional = True  # not used if tagger is non-RNN model


class BertABSATagger(BertPreTrainedModel):
    def __init__(self, bert_config):
        """

        :param bert_config: configuration for bert model
        """
        super(BertABSATagger, self).__init__(bert_config)
        self.num_labels = bert_config.num_labels
        self.tagger_config = TaggerConfig()
        self.tagger_config.absa_type = bert_config.absa_type.lower()
        if bert_config.tfm_mode == 'finetune':
            # initialized with pre-trained BERT and perform finetuning
            # print("Fine-tuning the pre-trained BERT...")
            self.bert = BertModel(bert_config)
        else:
            raise Exception("Invalid transformer mode %s!!!" % bert_config.tfm_mode)
        self.bert_dropout = nn.Dropout(bert_config.hidden_dropout_prob)
        # fix the parameters in BERT and regard it as feature extractor
        if bert_config.fix_tfm:
            # fix the parameters of the (pre-trained or randomly initialized) transformers during fine-tuning
            for p in self.bert.parameters():
                p.requires_grad = False

        self.tagger = None
        if self.tagger_config.absa_type == 'linear':
            # hidden size at the penultimate layer
            penultimate_hidden_size = bert_config.hidden_size
        else:
            self.tagger_dropout = nn.Dropout(self.tagger_config.hidden_dropout_prob)
            if self.tagger_config.absa_type == 'lstm':
                self.tagger = LSTM(input_size=bert_config.hidden_size,
                                   hidden_size=self.tagger_config.hidden_size,
                                   bidirectional=self.tagger_config.bidirectional)
            elif self.tagger_config.absa_type == 'gru':
                self.tagger = GRU(input_size=bert_config.hidden_size,
                                  hidden_size=self.tagger_config.hidden_size,
                                  bidirectional=self.tagger_config.bidirectional)
            elif self.tagger_config.absa_type == 'tfm':
                # transformer encoder layer
                self.tagger = nn.TransformerEncoderLayer(d_model=bert_config.hidden_size,
                                                         nhead=12,
                                                         dim_feedforward=4*bert_config.hidden_size,
                                                         dropout=0.1)
            elif self.tagger_config.absa_type == 'san':
                # vanilla self attention networks
                self.tagger = SAN(d_model=bert_config.hidden_size, nhead=12, dropout=0.1)
            elif self.tagger_config.absa_type == 'crf':
                self.tagger = CRF(num_tags=self.num_labels)
            else:
                raise Exception('Unimplemented downstream tagger %s...' % self.tagger_config.absa_type)
            penultimate_hidden_size = self.tagger_config.hidden_size
        self.classifier = nn.Linear(penultimate_hidden_size, bert_config.num_labels)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        # [! my change here! ] bert_outputs is a deep-copy of original bert output
        bert_outputs = outputs.clone()
        # the hidden states of the last Bert Layer, shape: (bsz, seq_len, hsz)
        tagger_input = outputs[0]
        tagger_input = self.bert_dropout(tagger_input)
        #print("tagger_input.shape:", tagger_input.shape)
        if self.tagger is None or self.tagger_config.absa_type == 'crf':
            # regard classifier as the tagger
            logits = self.classifier(tagger_input)
        else:
            if self.tagger_config.absa_type == 'lstm':
                # customized LSTM
                classifier_input, _ = self.tagger(tagger_input)
            elif self.tagger_config.absa_type == 'gru':
                # customized GRU
                classifier_input, _ = self.tagger(tagger_input)
            elif self.tagger_config.absa_type == 'san' or self.tagger_config.absa_type == 'tfm':
                # vanilla self-attention networks or transformer
                # adapt the input format for the transformer or self attention networks
                tagger_input = tagger_input.transpose(0, 1)
                classifier_input = self.tagger(tagger_input)
                classifier_input = classifier_input.transpose(0, 1)
            else:
                raise Exception("Unimplemented downstream tagger %s..." % self.tagger_config.absa_type)
            classifier_input = self.tagger_dropout(classifier_input)
            logits = self.classifier(classifier_input)
        outputs = (logits,) + outputs[2:]

        if labels is not None:
            if self.tagger_config.absa_type != 'crf':
                loss_fct = CrossEntropyLoss()
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = logits.view(-1, self.num_labels)[active_loss]
                    active_labels = labels.view(-1)[active_loss]
                    loss = loss_fct(active_logits, active_labels)
                else:
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                outputs = (loss,) + outputs
            else:
                log_likelihood = self.tagger(inputs=logits, tags=labels, mask=attention_mask)
                loss = -log_likelihood
                outputs = (loss,) + outputs
        return outputs, logits, bert_outputs


class DA_ABSA2Rec(nn.Module):
    def __init__(self, bert_config, Dataset_name, num_concats, ori_embedding_size, reduced_embedding_size, num_aspects):
        self.num_concats = num_concats
        self.ori_embedding_size = ori_embedding_size
        self.reduced_embedding_size = reduced_embedding_size
        self.num_aspects = num_aspects

        self.bertABSATagger = BertABSATagger(bert_config)
        self.aspect_categorys = nn.Embedding(num_aspects, reduced_embedding_size)

        self.embedding_to_rating = nn.Linear(in_features, out_features, bias=true)
        

    def embedding_layer(self):
        pass

    def aspect_category_attention(self):
        pass

    def sentiment_aware_attention(self):
        pass

    def compute_rating(self, user_embedding, item_embedding):
        concated_embedding = torch.concat(user_embedding, item_embedding)
        return self.embedding_to_rating(concated_embedding)

    def concat_bert_outputs(self, num_concats):
        pass

    def forward(self):
        pass