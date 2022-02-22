import torch
import torch.nn as nn
import os
from transformers import BertModel, XLNetModel
from seq_utils import *
from bert import BertPreTrainedModel, XLNetPreTrainedModel
from torch.nn import CrossEntropyLoss


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
            raise Exception("Invalid transformer mode %s!!!" %
                            bert_config.tfm_mode)
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
            self.tagger_dropout = nn.Dropout(
                self.tagger_config.hidden_dropout_prob)
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
                self.tagger = SAN(
                    d_model=bert_config.hidden_size, nhead=12, dropout=0.1)
            elif self.tagger_config.absa_type == 'crf':
                self.tagger = CRF(num_tags=self.num_labels)
            else:
                raise Exception('Unimplemented downstream tagger %s...' %
                                self.tagger_config.absa_type)
            penultimate_hidden_size = self.tagger_config.hidden_size
        self.classifier = nn.Linear(
            penultimate_hidden_size, bert_config.num_labels)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None, fine_tune=False):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        # the hidden states of the last Bert Layer, shape: (bsz, seq_len, hsz)
        tagger_input = outputs[0]
        last_hidden_state = tagger_input.clone()
        if fine_tune == False:
            last_hidden_state=last_hidden_state.detach()
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
                raise Exception("Unimplemented downstream tagger %s..." %
                                self.tagger_config.absa_type)
            classifier_input = self.tagger_dropout(classifier_input)
            logits = self.classifier(classifier_input)
        outputs = (logits,) + outputs[2:]

        if labels is not None:
            if self.tagger_config.absa_type != 'crf':
                loss_fct = CrossEntropyLoss()
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = logits.view(-1,
                                                self.num_labels)[active_loss]
                    active_labels = labels.view(-1)[active_loss]
                    loss = loss_fct(active_logits, active_labels)
                else:
                    loss = loss_fct(
                        logits.view(-1, self.num_labels), labels.view(-1))
                outputs = (loss,) + outputs
            else:
                log_likelihood = self.tagger(
                    inputs=logits, tags=labels, mask=attention_mask)
                loss = -log_likelihood
                outputs = (loss,) + outputs
        # output[0] = tagging label loss, shape: (n_gpus)xx
        # output[1] = tagging logits  shape: (n_gpus * batch_size, seq_len, num_labes=14)
        return outputs, last_hidden_state


class DA_ABSA2Rec_ori(nn.Module):
    def __init__(self, args, num_users, num_items, batch_size=None, max_sequence_length=512, ori_embedding_size=768, reduced_embedding_size=192, num_aspects=5):
        super(DA_ABSA2Rec_ori, self).__init__()
        self.args = args
        self.ori_embedding_size = ori_embedding_size
        self.reduced_embedding_size = reduced_embedding_size
        self.num_aspects = num_aspects
        self.batch_size = batch_size

        # self.bertABSATagger = BertABSATagger.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path),
        #                                                      config=bert_config, cache_dir='./cache')
        # fine_tuned_parameter_path = os.path.join('BERT_E2E_ABSA', 'bert-tfm-laptop14-finetune', 'pytorch_model.bin')
        # self.bertABSATagger = BertABSATagger.from_pretrained(fine_tuned_parameter_path, from_tf=False,
        #                             config=bert_config, cache_dir='./cache')
        self.aspect_categorys = nn.Embedding(
            num_aspects, reduced_embedding_size)
        self.aspect_categorys.weight.data.normal_(
            0, 0.01)  # 初始化需要学习的 aspect categorys 表示向量
        # self.pooler = nn.AdaptiveAvgPool1d(reduced_embedding_size)
        # self.pooler = nn.AdaptiveMaxPool1d(reduced_embedding_size)
        self.pooler = nn.Sequential(
            nn.Linear(in_features=self.ori_embedding_size, out_features=self.reduced_embedding_size),
            nn.Sigmoid(),
        )

        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.user_bias.weight.data.normal_(0, 0.1)  # 初始化 user 偏置
        self.user_bias.weight.data.normal_(0, 0.1)  # 初始化 item 偏置

        # 将tagging logits 转为情感注意力
        self.to_sentiment_score = nn.Sequential(
            nn.Linear(in_features=14, out_features=1),
            nn.Sigmoid()
        )
        self.to_sentiment_score[0].weight.data = torch.tensor([[0.0, 0.0, 0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0, -0.25, -0.25, -0.25, -0.25]], dtype=torch.float32, device=args.device)
        # self.to_sentiment_score = nn.Sequential(
        #     nn.Linear(in_features=14, out_features=14*self.num_aspects),
        #     nn.Dropout(0.1),
        #     nn.Linear(in_features=14*self.num_aspects, out_features=14),
        #     nn.Dropout(0.1),
        #     nn.Linear(in_features=14, out_features=1),
        #     nn.LeakyReLU()
        # )

        # 评分计算: 一层线性层的方式
        self.embedding_to_rating = nn.Sequential(
            nn.Linear(in_features=reduced_embedding_size, out_features=4 * reduced_embedding_size, bias=True),
            nn.Dropout(p=0.1),
            nn.Sigmoid(),
            nn.Linear(in_features=4 * reduced_embedding_size, out_features=reduced_embedding_size, bias=True),
            nn.Dropout(p=0.1),
            nn.Sigmoid(),
            nn.Linear(in_features=reduced_embedding_size, out_features=1, bias=True),
            nn.LeakyReLU()
        )
        for layer in self.embedding_to_rating:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0)


        # 评分计算: MLP 方式
        # self.embedding_to_rating = nn.Sequential(
        #     nn.Linear(in_features=reduced_embedding_size, out_features=2*reduced_embedding_size),
        #     nn.Linear(in_features=2*reduced_embedding_size, out_features=1)
        # )
        # 评分计算：类CATN方式:仅能用于用户/物品特征相互独立的方法
        # self.aspect_correlation_matrix = nn.Embedding(num_aspects, num_aspects)

    def ABSA_layer(self, input_ids, token_type_ids=None, attention_mask=None, tagging_labels=None,
                   position_ids=None, head_mask=None, fine_tune=False):

        return self.bertABSATagger(input_ids, position_ids=position_ids,
                                                 token_type_ids=token_type_ids, attention_mask=attention_mask, labels=tagging_labels, fine_tune=fine_tune)

        #tagging_loss, tagging_logits, last_hidden_state = absa_tagger_output[0], absa_tagger_output[1], absa_tagger_output[2]
        # last_hidden_state.shape : (batch_size, seq_len, ori_embedding_size)
        # return tagging_loss, tagging_logits, last_hidden_state

    def aspect_gate_control(self, last_hidden_state, attention_mask, tagging_labels):
        # tagging_labels.shape: (n_gpu * batch_size, seq_len)
        # attention_mask.shape(n_gpu * batch_size, seq_len)
        valid_seq_len = torch.count_nonzero(attention_mask, dim=-1)
        attention_mask = attention_mask.unsqueeze(dim=-1)
        last_hidden_state = last_hidden_state * attention_mask  # 不考虑 padding 的位置
        # 表示这个 batch 中每个序列中非 [PAD] 的有效序列的长度，shape: (batch_size)
        # label=0, 对应tag = O

        def tagging_label_to_mask(x, *y):
            if x == 0:
                return 0.05
            else:
                return 1
        mask_matrix = tagging_labels.clone()
#         mask_matrix = mask_matrix.to('cpu')
        map(tagging_label_to_mask, mask_matrix)
#         mask_matrix = mask_matrix.to(self.args.device)
        mask_matrix = mask_matrix.unsqueeze(
            dim=-1)  # shape (batch_size, seq_len,1)

        # 依靠 tensor 乘法的广播机制，会将(batch_size, seq_len,1)的maks矩阵广播成跟last_hidden_state一样的shape
        gate_control_output = last_hidden_state * mask_matrix

        # embedding 降维
        # shape: (batch_size, seq_len, redureduced_embedding_sizeced)
        gate_control_output = self.pooler(gate_control_output)
        return gate_control_output, valid_seq_len

    def aspect_category_attention(self, gate_control_output):
        aspect_specific_embeddings = []

        for aspect_index in range(self.num_aspects):
            broadcasted_aspect_category = self.aspect_categorys(torch.tensor([aspect_index]).to(self.args.device)).unsqueeze(
                dim=0)  # shape:(1,reduced_embedding_size)
            broadcasted_aspect_category = broadcasted_aspect_category.unsqueeze(
                dim=0)  # shape:(1, 1, reduced_embedding_size)

            # 先通过广播机制将 broadcasted_aspect_category 广播成 shape: (batch_size, seq_len, reduced_embedding_size), 再通过 sum 转成每个位置的内积
            # shape(batch_size, seq_len)
            dot_products = torch.sum(
                broadcasted_aspect_category * gate_control_output, dim=-1)
            temp_aspect_attention = nn.functional.softmax(
                dot_products, dim=-1)  # shape(batch_size, seq_len)
            temp_aspect_attention = temp_aspect_attention.unsqueeze(
                dim=-1)  # shape(batch_size, seq_len, -1) 后面点乘的时候直接广播
            aspect_specific_embeddings.append(
                gate_control_output * temp_aspect_attention)

        # shape(num_aspects, batch_size, seq_len, reduced_embedding_size)
        return aspect_specific_embeddings

    def sentiment_aware_attention(self, aspect_specific_embeddings, tagging_logits, valid_seq_len):
        # tagging_logits.shape: (batch_size, seq_len, 14)
        # valid_seq_len.shape: (batch_size)
        # sentiment_score.shape:  (batch_size, seq_len, 1)
        valid_seq_len = valid_seq_len.unsqueeze(dim=-1)
        sentiment_score = self.to_sentiment_score(nn.functional.softmax(tagging_logits, dim=-1))
        sentiment_attention = nn.functional.softmax(
            sentiment_score, dim=-1)  # shape:  (batch_size, seq_len, 1)
        
        for temp_aspect_specific_embedding in aspect_specific_embeddings:
            temp_aspect_specific_embedding = sentiment_attention * temp_aspect_specific_embedding
        
        # shape (num_aspects, batch_size, reduced_embedding_size) 所有位置的embedding加起来，融合成一个位置长度的 embedding
        sentiment_aware_embeddings = [] # list of len num_aspects, with each element shape:(batch_size, redueced_embedding_size)
        for temp_aspect_specific_embedding in aspect_specific_embeddings:
            temp_sentiment_aware_embedding = torch.sum(temp_aspect_specific_embedding, dim=2) / valid_seq_len
            sentiment_aware_embeddings.append(temp_sentiment_aware_embedding)
        sentiment_aware_embeddings = torch.stack(sentiment_aware_embeddings)

        return sentiment_aware_embeddings

    def compute_rating(self, sentiment_aware_embeddings, uids, iids):
        # review_embeddings (Tensor): (num_aspect, batch_size, reduced_embedding_size)
        aspect_specific_rating = self.embedding_to_rating(
            sentiment_aware_embeddings)  # shape: (num_aspect, batch_size)
        predicted_ratings = torch.sum(
            aspect_specific_rating, dim=0)/self.num_aspects   # shape: (batch_size)
        predicted_ratings = predicted_ratings + self.user_bias(uids) + self.item_bias(iids)
        predicted_ratings = 1 + 4 * torch.sigmoid(predicted_ratings) # 限制输出范围为 1~5
        
        return predicted_ratings.squeeze()
        # MLP 方式

    def forward(self, input_ids, uids, iids, token_type_ids=None, attention_mask=None, tagging_labels=None,
                position_ids=None, head_mask=None, fine_tune=False):

        (tagging_loss, tagging_logits), last_hidden_state = self.ABSA_layer(input_ids, token_type_ids=token_type_ids,
                                      attention_mask=attention_mask, tagging_labels=tagging_labels, fine_tune=fine_tune)
        if fine_tune == True:
            return tagging_loss
        gate_control_output, valid_seq_len = self.aspect_gate_control(
            last_hidden_state, attention_mask, tagging_labels)

        aspect_specific_embeddings = self.aspect_category_attention(
            gate_control_output)

        sentiment_aware_embeddings = self.sentiment_aware_attention(
            aspect_specific_embeddings, tagging_logits, valid_seq_len)
        
        predicted_ratings = self.compute_rating(sentiment_aware_embeddings, uids, iids)

        return predicted_ratings

class DA_ABSA2Rec_new(nn.Module):
    def __init__(self, args, num_users, num_items, max_sequence_length=512, ori_embedding_size=768, reduced_embedding_size=192, num_aspects=5):
        super(DA_ABSA2Rec_new, self).__init__()
        self.args = args
        self.max_sequence_length = max_sequence_length
        self.ori_embedding_size = ori_embedding_size
        self.reduced_embedding_size = reduced_embedding_size
        self.num_aspects = num_aspects

        self.aspect_categorys = nn.Embedding(
            num_aspects, reduced_embedding_size)
        self.aspect_categorys.weight.data.normal_(
            0, 0.01)  # 初始化需要学习的 aspect categorys 表示向量
        # self.pooler = nn.AdaptiveAvgPool1d(reduced_embedding_size)
        self.pooler = nn.AdaptiveMaxPool1d(reduced_embedding_size)
        # self.pooler = nn.Sequential(
        #     nn.Linear(in_features=self.ori_embedding_size, out_features=self.reduced_embedding_size),
        #     nn.Sigmoid(),
        # )

        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.user_bias.weight.data.normal_(0, 0.1)  # 初始化 user 偏置
        self.user_bias.weight.data.normal_(0, 0.1)  # 初始化 item 偏置

        self.concat_to_1 = nn.Sequential(
            nn.Linear(2*self.reduced_embedding_size, 1)
        )
        nn.init.xavier_normal_(self.concat_to_1[0].weight)
        nn.init.constant_(self.concat_to_1[0].bias, 0)

        # 将tagging logits 转为情感注意力
        self.to_sentiment_score = nn.Sequential(
            nn.Linear(in_features=14, out_features=1),
            nn.Sigmoid()
        )
        self.to_sentiment_score[0].weight.data = torch.tensor([[0.0, 0.0, 0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0, -0.25, -0.25, -0.25, -0.25]], dtype=torch.float32, device=args.device)
        # self.to_sentiment_score = nn.Sequential(
        #     nn.Linear(in_features=14, out_features=14*self.num_aspects),
        #     nn.Dropout(0.1),
        #     nn.Linear(in_features=14*self.num_aspects, out_features=14),
        #     nn.Dropout(0.1),
        #     nn.Linear(in_features=14, out_features=1),
        #     nn.LeakyReLU()
        # )

        # 评分计算: 一层线性层的方式
        self.embedding_to_rating = nn.Sequential(
            nn.Linear(in_features=reduced_embedding_size, out_features=4 * reduced_embedding_size, bias=True),
            nn.Dropout(p=0.1),
            nn.Sigmoid(),
            nn.Linear(in_features=4 * reduced_embedding_size, out_features=reduced_embedding_size, bias=True),
            nn.Dropout(p=0.1),
            nn.Sigmoid(),
            nn.Linear(in_features=reduced_embedding_size, out_features=1, bias=True),
            nn.LeakyReLU()
        )
        for layer in self.embedding_to_rating:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0)


        # 评分计算: MLP 方式
        # self.embedding_to_rating = nn.Sequential(
        #     nn.Linear(in_features=reduced_embedding_size, out_features=2*reduced_embedding_size),
        #     nn.Linear(in_features=2*reduced_embedding_size, out_features=1)
        # )
        # 评分计算：类CATN方式:仅能用于用户/物品特征相互独立的方法
        # self.aspect_correlation_matrix = nn.Embedding(num_aspects, num_aspects)

    def aspect_category_attention(self, gate_control_output):
        aspect_specific_embeddings = []

        for aspect_index in range(self.num_aspects):
            broadcasted_aspect_category = self.aspect_categorys(torch.tensor([aspect_index]).to(self.args.device)).unsqueeze(
                dim=0)  # shape:(1,reduced_embedding_size)
            broadcasted_aspect_category = broadcasted_aspect_category.unsqueeze(
                dim=0)  # shape:(1, 1, reduced_embedding_size)

            # 先通过广播机制将 broadcasted_aspect_category 广播成 shape: (batch_size, seq_len, reduced_embedding_size), 再通过 sum 转成每个位置的内积
            # shape(batch_size, seq_len)
            dot_products = torch.sum(
                broadcasted_aspect_category * gate_control_output, dim=-1)
            temp_aspect_attention = nn.functional.softmax(
                dot_products, dim=-1)  # shape(batch_size, seq_len)
            temp_aspect_attention = temp_aspect_attention.unsqueeze(
                dim=-1)  # shape(batch_size, seq_len, -1) 后面点乘的时候直接广播
            aspect_specific_embeddings.append(
                gate_control_output * temp_aspect_attention)

        # shape(num_aspects, batch_size, seq_len, reduced_embedding_size)
        return aspect_specific_embeddings

    def sentiment_aware_attention(self, aspect_specific_embeddings, tagging_logits):
        # tagging_logits.shape: (batch_size, seq_len, 14)
        # valid_seq_len.shape: (batch_size)
        # sentiment_score.shape:  (batch_size, seq_len, 1)
        sentiment_score = self.to_sentiment_score(nn.functional.softmax(tagging_logits, dim=-1))
        sentiment_attention = nn.functional.softmax(
            sentiment_score, dim=-1)  # shape:  (batch_size, seq_len, 1)
        
        for temp_aspect_specific_embedding in aspect_specific_embeddings:
            temp_aspect_specific_embedding = sentiment_attention * temp_aspect_specific_embedding

        # shape (num_aspects, batch_size, reduced_embedding_size) 所有位置的embedding加起来，融合成一个位置长度的 embedding
        sentiment_aware_embeddings = [] # list of len num_aspects, with each element shape:(batch_size, redueced_embedding_size)
        for temp_aspect_specific_embedding in aspect_specific_embeddings:
            # temp_sentiment_aware_embedding = torch.sum(temp_aspect_specific_embedding, dim=2) / self.max_sequence_length
            temp_sentiment_aware_embedding = torch.sum(temp_aspect_specific_embedding, dim=2)
            sentiment_aware_embeddings.append(temp_sentiment_aware_embedding)
        sentiment_aware_embeddings = torch.stack(sentiment_aware_embeddings)

        return sentiment_aware_embeddings

    def compute_rating(self, u_emb, i_emb, uids, iids):
        # review_embeddings (Tensor): (num_aspect, batch_size, reduced_embedding_size)
        # aspect_specific_rating = self.embedding_to_rating(
        #     sentiment_aware_embeddings)  # shape: (num_aspect, batch_size)
        # predicted_ratings = torch.sum(torch.sum(u_emb*i_emb, dim=-1), dim=0)/self.num_aspects   # shape: (batch_size)
        predicted_ratings = torch.sum(self.concat_to_1(torch.cat((u_emb, i_emb), dim=-1)), dim=0)/self.num_aspects 
        predicted_ratings = predicted_ratings.squeeze() + self.user_bias(uids).squeeze() + self.item_bias(iids).squeeze()
        predicted_ratings = 1 + 4 * torch.sigmoid(predicted_ratings) # 限制输出范围为 1~5

        return predicted_ratings.squeeze()
        # MLP 方式

    def forward(self, uid, user_emb, user_logits, iid, item_emb, item_logits):
        u_emb = self.pooler(user_emb)
        u_emb = self.aspect_category_attention(u_emb)
        u_emb = self.sentiment_aware_attention(u_emb, user_logits)

        i_emb = self.pooler(item_emb)
        i_emb = self.aspect_category_attention(i_emb)
        i_emb = self.sentiment_aware_attention(i_emb, item_logits)

        predicted_ratings = self.compute_rating(u_emb, i_emb, uid, iid)

        return predicted_ratings

class DA_ABSA2Rec_linear(nn.Module):
    def __init__(self, args, bert_config, num_users, num_items, batch_size=None, max_sequence_length=512, ori_embedding_size=768, reduced_embedding_size=192, num_aspects=3):
        super(DA_ABSA2Rec, self).__init__()
        self.args = args
        self.ori_embedding_size = ori_embedding_size
        self.reduced_embedding_size = reduced_embedding_size
        self.num_aspects = num_aspects
        self.batch_size = batch_size

        # self.bertABSATagger = BertABSATagger.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path),
        #                                                      config=bert_config, cache_dir='./cache')
        fine_tuned_parameter_path = os.path.join('BERT_E2E_ABSA', 'bert-tfm-laptop14-finetune', 'pytorch_model.bin')
        self.bertABSATagger = BertABSATagger.from_pretrained(fine_tuned_parameter_path, from_tf=False,
                                    config=bert_config, cache_dir='./cache')
        self.aspect_categorys = nn.Embedding(
            num_aspects, reduced_embedding_size)
        self.aspect_categorys.weight.data.normal_(
            0, 0.01)  # 初始化需要学习的 aspect categorys 表示向量
        # self.pooler = nn.AdaptiveAvgPool1d(reduced_embedding_size)
        # self.pooler = nn.AdaptiveMaxPool1d(reduced_embedding_size)
        self.pooler = nn.Sequential(
            nn.Linear(in_features=self.ori_embedding_size, out_features=self.reduced_embedding_size),
            nn.Sigmoid(),
        )

        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.user_bias.weight.data.normal_(0, 0.1)  # 初始化 user 偏置
        self.user_bias.weight.data.normal_(0, 0.1)  # 初始化 item 偏置

        # 将tagging logits 转为情感注意力
        self.to_sentiment_score = nn.Sequential(
            nn.Linear(in_features=14, out_features=1),
            nn.Sigmoid()
        )
        self.to_sentiment_score[0].weight.data = torch.tensor([[0.0, 0.0, 0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0, -0.25, -0.25, -0.25, -0.25]], dtype=torch.float32, device=args.device)
        # self.to_sentiment_score = nn.Sequential(
        #     nn.Linear(in_features=14, out_features=14*self.num_aspects),
        #     nn.Dropout(0.1),
        #     nn.Linear(in_features=14*self.num_aspects, out_features=14),
        #     nn.Dropout(0.1),
        #     nn.Linear(in_features=14, out_features=1),
        #     nn.LeakyReLU()
        # )

        self.embedding_to_rating = nn.Sequential(
            nn.Linear(in_features=reduced_embedding_size, out_features=4 * reduced_embedding_size, bias=True),
            nn.Dropout(p=0.1),
            nn.Sigmoid(),
            nn.Linear(in_features=4 * reduced_embedding_size, out_features=reduced_embedding_size, bias=True),
            nn.Dropout(p=0.1),
            nn.Sigmoid(),
            nn.Linear(in_features=reduced_embedding_size, out_features=1, bias=True),
            nn.LeakyReLU()
        )
        self.to_reduced = nn.Linear(ori_embedding_size, reduced_embedding_size)
        nn.init.xavier_normal_(to_reduced.weight)
        nn.init.constant_(to_reduced.bias, 0)

        self.ori_to_1 = nn.Linear(ori_embedding_size, 1)
        nn.init.xavier_normal_(ori_to_1.weight)
        nn.init.constant_(ori_to_1.bias, 0)

        self.redueced_to_1 = nn.Linear(reduced_embedding_size, 1)
        nn.init.xavier_normal_(redueced_to_1.weight)
        nn.init.constant_(redueced_to_1.bias, 0)

        self.position_to_1 = nn.Linear(max_sequence_length, 1)
        nn.init.xavier_normal_(position_to_1.weight)
        nn.init.constant_(position_to_1.bias, 0)

        for layer in self.embedding_to_rating:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0)

    def ABSA_layer(self, input_ids, token_type_ids=None, attention_mask=None, tagging_labels=None,
                   position_ids=None, head_mask=None, fine_tune=False):

        return self.bertABSATagger(input_ids, position_ids=position_ids,
                                                 token_type_ids=token_type_ids, attention_mask=attention_mask, labels=tagging_labels, fine_tune=fine_tune)

        #tagging_loss, tagging_logits, last_hidden_state = absa_tagger_output[0], absa_tagger_output[1], absa_tagger_output[2]
        # last_hidden_state.shape : (batch_size, seq_len, ori_embedding_size)
        # return tagging_loss, tagging_logits, last_hidden_state

    def aspect_gate_control(self, last_hidden_state, attention_mask=None, tagging_labels=None):
        # tagging_labels.shape: (n_gpu * batch_size, seq_len)
        # attention_mask.shape(n_gpu * batch_size, seq_len)
        if attention_mask!= None:
            valid_seq_len = torch.count_nonzero(attention_mask, dim=-1)
            attention_mask = attention_mask.unsqueeze(dim=-1)
            last_hidden_state = last_hidden_state * attention_mask  # 不考虑 padding 的位置
        # 表示这个 batch 中每个序列中非 [PAD] 的有效序列的长度，shape: (batch_size)
        # label=0, 对应tag = O

        def tagging_label_to_mask(x, *y):
            if x == 0:
                return 0.05
            else:
                return 1
        mask_matrix = tagging_labels.clone()
#         mask_matrix = mask_matrix.to('cpu')
        map(tagging_label_to_mask, mask_matrix)
#         mask_matrix = mask_matrix.to(self.args.device)
        mask_matrix = mask_matrix.unsqueeze(
            dim=-1)  # shape (batch_size, seq_len,1)

        # 依靠 tensor 乘法的广播机制，会将(batch_size, seq_len,1)的maks矩阵广播成跟last_hidden_state一样的shape
        gate_control_output = last_hidden_state * mask_matrix

        # embedding 降维
        # shape: (batch_size, seq_len, redureduced_embedding_sizeced)
        gate_control_output = self.pooler(gate_control_output)
        return gate_control_output, valid_seq_len

    def aspect_category_attention(self, gate_control_output):
        aspect_specific_embeddings = []

        for aspect_index in range(self.num_aspects):
            broadcasted_aspect_category = self.aspect_categorys(torch.tensor([aspect_index]).to(self.args.device)).unsqueeze(
                dim=0)  # shape:(1,reduced_embedding_size)
            broadcasted_aspect_category = broadcasted_aspect_category.unsqueeze(
                dim=0)  # shape:(1, 1, reduced_embedding_size)

            # 先通过广播机制将 broadcasted_aspect_category 广播成 shape: (batch_size, seq_len, reduced_embedding_size), 再通过 sum 转成每个位置的内积
            # shape(batch_size, seq_len)
            dot_products = torch.sum(
                broadcasted_aspect_category * gate_control_output, dim=-1)
            temp_aspect_attention = nn.functional.softmax(
                dot_products, dim=-1)  # shape(batch_size, seq_len)
            temp_aspect_attention = temp_aspect_attention.unsqueeze(
                dim=-1)  # shape(batch_size, seq_len, -1) 后面点乘的时候直接广播
            aspect_specific_embeddings.append(
                gate_control_output * temp_aspect_attention)

        # shape(num_aspects, batch_size, seq_len, reduced_embedding_size)
        return aspect_specific_embeddings

    def sentiment_aware_attention(self, aspect_specific_embeddings, tagging_logits, valid_seq_len=None):
        # tagging_logits.shape: (batch_size, seq_len, 14)
        # valid_seq_len.shape: (batch_size)
        # sentiment_score.shape:  (batch_size, seq_len, 1)
        valid_seq_len = valid_seq_len.unsqueeze(dim=-1)
        sentiment_score = self.to_sentiment_score(tagging_logits)
        sentiment_attention = nn.functional.softmax(
            sentiment_score, dim=-1)  # shape:  (batch_size, seq_len, 1)

        for i in range(len(aspect_specific_embeddings)):
            aspect_specific_embeddings[i] = sentiment_attention * aspect_specific_embeddings[i]

        # 所有位置的embedding加起来，融合成一个位置长度的 embedding
        sentiment_aware_embeddings = [] # list of len num_aspects, with each element shape:(batch_size, redueced_embedding_size)
        for temp_aspect_specific_embedding in aspect_specific_embeddings:
            # temp_sentiment_aware_embedding = torch.sum(temp_aspect_specific_embedding, dim=2) / valid_seq_len
            temp_sentiment_aware_embedding = torch.sum(temp_aspect_specific_embedding, dim=2)
            sentiment_aware_embeddings.append(temp_sentiment_aware_embedding)

        # shape (num_aspects, batch_size, reduced_embedding_size) 
        sentiment_aware_embeddings = torch.stack(sentiment_aware_embeddings)

        return sentiment_aware_embeddings

    def compute_rating(self, sentiment_aware_embeddings, uids, iids):
        # review_embeddings (Tensor): (num_aspect, batch_size, reduced_embedding_size)
        # 一层线性层的方式
        aspect_specific_rating = self.embedding_to_rating(
            sentiment_aware_embeddings)  # shape: (num_aspect, batch_size)
        predicted_ratings = torch.sum(
            aspect_specific_rating, dim=0)/self.num_aspects   # shape: (batch_size)
        predicted_ratings = predicted_ratings + self.user_bias(uids) + self.item_bias(iids)
        predicted_ratings = 1 + 4 * torch.sigmoid(predicted_ratings) # 限制输出范围为 1~5
        
        return predicted_ratings.squeeze()
        # MLP 方式

    def forward(self, input_ids, uids, iids, token_type_ids=None, attention_mask=None, tagging_labels=None,
                position_ids=None, head_mask=None, fine_tune=False):

        (tagging_loss, tagging_logits), last_hidden_state = self.ABSA_layer(input_ids, token_type_ids=token_type_ids,
                                      attention_mask=attention_mask, tagging_labels=tagging_labels, fine_tune=fine_tune)
        if fine_tune == True:
            return tagging_loss

        # shape: (batch_size, seq_len)
        # sentiment_attention = nn.functional.softmax(self.to_sentiment_score(nn.functional.softmax(tagging_logits, dim=-1)), dim=-1).squeeze()

        # shape: (batch_size, seq_len, reduced_embedding_size)
        output = torch.sigmoid(self.ori_to_1(last_hidden_state)).squeeze()

        # shape: (batch_size, seq_len)
        # output = torch.sigmoid(self.redueced_to_1(output)).squeeze() * sentiment_attention
        output = torch.sigmoid(self.redueced_to_1(output)).squeeze()

        predicted_ratings = nn.functional.leaky_relu(self.position_to_1(output)).squeeze()
        # predicted_ratings = self.position_to_1(output).squeeze()

        return predicted_ratings


class DA_ABSA2Rec(nn.Module):
    def __init__(self, args, num_users, num_items, max_sequence_length=512, ori_embedding_size=768, reduced_embedding_size=128, h2=50, num_aspects=5):
        super(DA_ABSA2Rec, self).__init__()
        self.args = args
        self.max_sequence_length = max_sequence_length
        self.ori_embedding_size = ori_embedding_size
        self.reduced_embedding_size = reduced_embedding_size
        self.num_aspects = num_aspects
        self.h2 = h2

        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        # self.user_bias.weight.data.normal_(0, 0.01)  # 初始化 user 偏置
        # self.item_bias.weight.data.normal_(0, 0.01)  # 初始化 item 偏置
        self.user_bias.weight.data.uniform_(-0.01, 0.01)  # 初始化 user 偏置
        self.item_bias.weight.data.uniform_(-0.01, 0.01)  # 初始化 item 偏置

        self.aspect_categorys = nn.Embedding(
            num_aspects, reduced_embedding_size)
        self.aspect_categorys.weight.data.normal_(
            0, 0.01)

        # 将tagging logits 转为情感注意力
        self.to_sentiment_score = nn.Sequential(
            nn.Linear(in_features=14, out_features=1),
            nn.Sigmoid()
        )
        self.pooler = nn.AdaptiveMaxPool1d(reduced_embedding_size)
        self.pooler = nn.AdaptiveAvgPool1d(reduced_embedding_size)
        self.to_sentiment_score[0].weight.data = torch.tensor([[0.0, 0.0, 0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0, -0.25, -0.25, -0.25, -0.25]], dtype=torch.float32, device=args.device)
        nn.init.constant_(self.to_sentiment_score[0].bias, 0)
        # self.to_sentiment_score = nn.Sequential(
        #     nn.Linear(in_features=14, out_features=14*self.num_aspects),
        #     nn.Dropout(0.1),
        #     nn.Sigmoid(),
        #     nn.Linear(in_features=14*self.num_aspects, out_features=14),
        #     nn.Dropout(0.1),
        #     nn.Sigmoid(),
        #     nn.Linear(in_features=14, out_features=1),
        #     nn.LeakyReLU()
        # )
        # nn.init.xavier_normal_(self.to_sentiment_score[0].weight)
        # nn.init.constant_(self.to_sentiment_score[0].bias, 0)
        # nn.init.xavier_normal_(self.to_sentiment_score[3].weight)
        # nn.init.constant_(self.to_sentiment_score[3].bias, 0)
        # nn.init.xavier_normal_(self.to_sentiment_score[6].weight)
        # nn.init.constant_(self.to_sentiment_score[6].bias, 0)

        self.to_reduced = nn.Sequential(
            nn.Linear(ori_embedding_size, reduced_embedding_size),
            # nn.BatchNorm1d(self.max_sequence_length)
            nn.Dropout(0.5),
            nn.Sigmoid()
        )
        nn.init.xavier_normal_(self.to_reduced[0].weight)
        # nn.init.uniform_(self.to_reduced[0].weight, -0.01, 0.01)
        nn.init.constant_(self.to_reduced[0].bias, 0)

        # self.ori_to_1 = nn.Linear(ori_embedding_size, 1)
        self.redueced_to_1 = nn.Sequential(
            nn.Linear(reduced_embedding_size, 1),
            # nn.BatchNorm1d(self.max_sequence_length)
            nn.Dropout(0.5)
        )
        nn.init.xavier_normal_(self.redueced_to_1[0].weight)
        # nn.init.uniform_(self.redueced_to_1[0].weight, -0.01, 0.01)
        nn.init.constant_(self.redueced_to_1[0].bias, 0)

        self.position_to_1 = nn.Sequential(
            nn.Linear(max_sequence_length, 1),
        )
        nn.init.xavier_normal_(self.position_to_1[0].weight)
        # nn.init.uniform_(self.position_to_1[0].weight, -0.01, 0.01)
        nn.init.constant_(self.position_to_1[0].bias, 0)

        self.concat_to_1 = nn.Sequential(
            # nn.Linear(2*self.reduced_embedding_size, 4*reduced_embedding_size),
            # nn.Dropout(0.5),
            # nn.Sigmoid(),
            # nn.Linear(4*reduced_embedding_size, 2*reduced_embedding_size),
            # nn.Dropout(0.5),
            # nn.Sigmoid(),
            nn.Linear(2*reduced_embedding_size, 1),
            nn.Sigmoid(),
            # nn.BatchNorm1d(1)
        )
        nn.init.xavier_normal_(self.concat_to_1[0].weight)
        # nn.init.uniform_(self.concat_to_1[0].weight, -0.01, 0.01)
        nn.init.constant_(self.concat_to_1[0].bias, 0)
        # nn.init.xavier_normal_(self.concat_to_1[3].weight)
        # nn.init.constant_(self.concat_to_1[3].bias, 0)
        # nn.init.xavier_normal_(self.concat_to_1[6].weight)
        # nn.init.constant_(self.concat_to_1[6].bias, 0)

        self.concat_to_1_s = nn.Sequential(
            nn.Linear(2*14, 1),
            nn.Sigmoid()
        )
        nn.init.xavier_normal_(self.concat_to_1_s[0].weight)
        # nn.init.uniform_(self.concat_to_1_s[0].weight, -0.01, 0.01)
        nn.init.constant_(self.concat_to_1_s[0].bias, 0)

        # # below are weight matrix/vectors needed for ANR's rating prediction method 
        # self.W_a = nn.Parameter(torch.Tensor(self.num_aspects, self.num_aspects), requires_grad = True)
        # self.W_s = nn.Parameter(torch.Tensor(self.reduced_embedding_size, self.reduced_embedding_size), requires_grad = True)
        # self.W_s.data.normal_(0, 0.01)

        # self.W_u = nn.Parameter(torch.Tensor(self.reduced_embedding_size, self.h2), requires_grad=True)
        # self.w_hu = nn.Parameter(torch.Tensor(self.h2, 1), requires_grad = True)
        # self.W_i = nn.Parameter(torch.Tensor(self.reduced_embedding_size, self.h2), requires_grad=True)
        # self.w_hi = nn.Parameter(torch.Tensor(self.h2, 1), requires_grad = True)

        # nn.init.kaiming_normal_(self.W_u)
        # nn.init.kaiming_normal_(self.W_i)
        # nn.init.kaiming_normal_(self.w_hu)
        # nn.init.kaiming_normal_(self.w_hi)

        # self.tagger = nn.TransformerEncoderLayer(d_model=self.reduced_embedding_size,
        #                                          nhead=8,
        #                                          dim_feedforward=4*reduced_embedding_size,
        #                                          dropout=0.1,
        #                                          batch_first=True)

        # self.classifier = nn.Linear(self.reduced_embedding_size, 14)
        # you may need to change the param_dict_path here
        # param_dict_path = os.path.join('BERT_E2E_ABSA', 'bert-tfm-laptop14-finetune', 'absa_param_dict.bin')
        # param_dict = torch.load(param_dict_path)
        # self.classifier.weight.data = param_dict['weight']
        # self.classifier.bias.data = param_dict['bias']
        # print('loaded the absa_param_dict')

    def aspect_category_attention(self, embedding):
        aspect_specific_embeddings = []

        for aspect_index in range(self.num_aspects):
            # 先通过广播机制将 broadcasted_aspect_category 广播成 shape: (batch_size, seq_len, reduced_embedding_size),
            broadcasted_aspect_category = self.aspect_categorys(torch.tensor([aspect_index]).to(self.args.device)).unsqueeze(
                dim=0)  # shape:(1,reduced_embedding_size)

            # 再通过 sum 转成每个位置的内积
            # shape(batch_size, seq_len)
            dot_products = torch.sum(broadcasted_aspect_category * embedding, dim=-1)
            temp_aspect_attention = nn.functional.softmax(dot_products, dim=-1)  # shape(batch_size, seq_len)
            temp_aspect_attention = temp_aspect_attention.unsqueeze(dim=-1)  # shape(batch_size, seq_len, 1) 后面点乘的时候直接广播
            aspect_specific_embeddings.append(torch.sum(embedding * temp_aspect_attention, dim=-2)) # shape(batch_size, reduced_embedding_size)

        # shape: (num_aspects, batch_size, reduced_embedding_size)
        aspect_specific_embeddings = torch.stack(aspect_specific_embeddings)
        return aspect_specific_embeddings

    def sentiment_aware_attention(self, aspect_specific_embeddings, tagging_logits, valid_seq_len=None):
        # tagging_logits.shape: (batch_size, seq_len, 14)
        # valid_seq_len.shape: (batch_size)
        # sentiment_score.shape:  (batch_size, seq_len, 1)
        # valid_seq_len = valid_seq_len.unsqueeze(dim=-1)
        sentiment_score = self.to_sentiment_score(tagging_logits)
        sentiment_attention = nn.functional.softmax(
            sentiment_score, dim=-1).unsqueeze(dim=0)  # shape:  (1, batch_size, seq_len, 1)

        aspect_specific_embeddings = sentiment_attention * aspect_specific_embeddings
        # 所有位置的embedding加起来，融合成一个位置长度的 embedding
        sentiment_aware_embeddings = torch.sum(aspect_specific_embeddings, dim=-2)
        # shape (num_aspects, batch_size, reduced_embedding_size) 

        return sentiment_aware_embeddings

    def compute_rating(self, u_emb, i_emb, uids, iids):
        # u_emb/i_emb (Tensor): (num_aspect, batch_size, reduced_embedding_size)

        # predicted_ratings = torch.sum(torch.sum(u_emb*i_emb, dim=-1), dim=0)/self.num_aspects   # shape: (batch_size)
        predicted_ratings = torch.sum(self.concat_to_1(torch.cat((u_emb, i_emb), dim=-1)).squeeze(), dim=0)/self.num_aspects
        predicted_ratings = 1 + 4 * torch.sigmoid(predicted_ratings) + self.user_bias(uids).squeeze() + self.item_bias(iids).squeeze() # 限制输出范围为 1~5

        # # below are ANR's methods for rating prediction, unfortunately we found it preforms not well, so we decide to use our simple way for rating prediction
        # u_emb = u_emb.permute(1,0,2) # shape: (batch_size, num_aspect, reduced_embedding_size)
        # i_emb = i_emb.permute(1,0,2) # shape: (batch_size, num_aspect, reduced_embedding_size)

        # u_emb_t = torch.transpose(u_emb, 1, 2) # shape: (batch_size, reduced_embedding_size, num_aspect)
        # i_emb_t = torch.transpose(i_emb, 1, 2) # shape: (batch_size, reduced_embedding_size, num_aspect)

        # S = torch.relu(torch.matmul(torch.matmul(u_emb, self.W_s), i_emb_t))# shape: (batch_size, num_aspect, num_aspect)

        # H_u = torch.relu(torch.matmul(u_emb, self.W_u) + torch.matmul(torch.transpose(S, 1, 2), torch.matmul(i_emb, self.W_i))) # (batch_size, num_aspect, h2)
        # beta_u = torch.softmax(torch.matmul(H_u, self.w_hu).squeeze(), dim=-1) # (batch_size, num_aspect)

        # H_i = torch.relu(torch.matmul(i_emb, self.W_i) + torch.matmul(S, torch.matmul(u_emb, self.W_u)))
        # beta_i = torch.softmax(torch.matmul(H_i, self.w_hi).squeeze(), dim=-1)

        # beta = beta_u * beta_i

        # predicted_ratings = torch.sum(u_emb*i_emb, dim=-1) # (batch_size, num_aspect)
        # predicted_ratings = torch.sum(beta * predicted_ratings, dim=-1) # (batch_size)
        # predicted_ratings = 1 + 4 * torch.sigmoid(predicted_ratings) + self.user_bias(uids).squeeze() + self.item_bias(iids).squeeze()

        return predicted_ratings

    def compute_tag_logits(self, emb):
        # classifier_input = self.tagger(emb)
        classifier_input = emb
        tag_logits = self.classifier(classifier_input)
        # tag_logits = nn.functional.dropout(tag_logits, p=0.1)

        return tag_logits

    def forward(self, uid, user_emb, user_logits, iid, item_emb, item_logits):
        # semtiment ratings
        # user_logits = self.compute_tag_logits(user_emb)
        # item_logits = self.compute_tag_logits(item_emb)

        user_prob = nn.functional.softmax(user_logits, dim=-1)
        item_prob = nn.functional.softmax(item_logits, dim=-1)
        user_emb_s = torch.sum(user_prob, dim=-2)
        item_emb_s = torch.sum(item_prob, dim=-2)

        # sentiment_ratings = torch.sum(user_emb_s*item_emb_s, dim=-1)
        sentiment_ratings = self.concat_to_1_s(torch.cat((user_emb_s, item_emb_s), dim=-1)).squeeze() + self.user_bias(uid).squeeze() + self.item_bias(iid).squeeze()
        sentiment_ratings = 1 + 4 * torch.sigmoid(sentiment_ratings)

        # no aspect and no sentiment
        user_output = user_emb
        item_output = item_emb

        user_output = torch.sum(user_emb, dim=-2) / self.max_sequence_length
        item_output = torch.sum(item_emb, dim=-2) / self.max_sequence_length

        # predicted_ratings = torch.sum(user_output*item_output, dim=-1)
        predicted_ratings = self.concat_to_1(torch.cat((user_output, item_output), dim=-1)).squeeze() + self.user_bias(uid).squeeze() + self.item_bias(iid).squeeze()
        predicted_ratings = 1 + 4 * torch.sigmoid(predicted_ratings)

        # # aspect + sentiment (or only one of them), we shall not use this since we found it not helpful to the performence
        # user_output = user_emb
        # item_output = item_emb

        # user_output = self.aspect_category_attention(user_output)
        # item_output = self.aspect_category_attention(item_output)

        # # user_output = user_output.unsqueeze(dim=2).expand(-1, -1, self.max_sequence_length, -1) # shape: (num_aspects, batch_size, max_seq_len, reduced_embedding_size) 
        # # item_output = item_output.unsqueeze(dim=2).expand(-1, -1, self.max_sequence_length, -1)

        # # user_output = self.sentiment_aware_attention(user_output, user_logits)
        # # item_output = self.sentiment_aware_attention(item_output, item_logits)

        # # shape:(batch_size)
        # predicted_ratings = self.compute_rating(user_output, item_output, uid, iid)

        return predicted_ratings, sentiment_ratings, user_output, item_output

# class Google_version(nn.Module):
#     def __init__(self, num_users, num_items, vocab_size, wid_wEmbed_path, user_doc_path, item_doc_path):
#         self.num_users = num_users
# 		self.num_items = num_items
#         self.vocab_size = vocab_size

# 		# User Documents & Item Documents (Input)
# 		self.uid_userDoc = nn.Embedding(self.num_users, self.512)
# 		self.uid_userDoc.weight.requires_grad = False
#         np_uid_userDoc = np.load( uid_userDoc_path )
#         self.uid_userDoc.weight.data.copy_(torch.from_numpy(np_uid_userDoc).long())

# 		self.iid_itemDoc = nn.Embedding(self.num_items, self.512)
# 		self.iid_itemDoc.weight.requires_grad = False
#         np_iid_itemDoc = np.load( iid_itemDoc_path )
#         self.iid_itemDoc.weight.data.copy_(torch.from_numpy(np_iid_itemDoc).long())

# 		# Word Embeddings (Input)
# 		self.wid_wEmbed = nn.Embedding(self.vocab_size, self.args.word_embed_dim)
# 		self.wid_wEmbed.weight.requires_grad = False
#         np_wid_wEmbed = np.load( wid_wEmbed_path )
#         self.wid_wEmbed.weight.data.copy_(torch.from_numpy(np_wid_wEmbed))

# 		# Aspect Representation Learning - Single Aspect-based Attention Network (Shared between User & Item)
# 		self.pooler = nn.AdaptiveAvgPool1d(128)
# 		self.concat_to_1 = nn.Sequential(
# 			nn.Linear(2*128, 1),
# 			nn.Sigmoid(),
# 		)
# 		nn.init.xavier_normal_(self.concat_to_1[0].weight)
# 		nn.init.constant_(self.concat_to_1[0].bias, 0)
    
#     def forward(self, uid, iid):
# 		# Input
# 		batch_userDoc = self.uid_userDoc(batch_uid)
# 		batch_itemDoc = self.iid_itemDoc(batch_iid)

# 		# Embedding Layer
# 		# shape: (bsz, seq_len, 300)
# 		batch_userDocEmbed = self.wid_wEmbed(batch_userDoc.long())
# 		batch_itemDocEmbed = self.wid_wEmbed(batch_itemDoc.long())

# 		# shape: (bsz, seq_len, 128)
# 		batch_userDocEmbed = self.pooler(batch_userDocEmbed)
# 		batch_itemDocEmbed = self.pooler(batch_itemDocEmbed)

# 		# shape: (bsz, 128)
# 		batch_userDocEmbed = torch.sum(batch_userDocEmbed, dim=-2)
# 		batch_itemDocEmbed = torch.sum(batch_itemDocEmbed, dim=-2)

# 		# shape: (bsz)
# 		rating_pred = self.concat_to_1(torch.cat((batch_userDocEmbed, batch_itemDocEmbed), dim=-1)).squeeze()

# 		return rating_pred