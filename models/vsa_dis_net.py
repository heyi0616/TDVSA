import torch
from torch import nn

from data import consts
from models import video_cnn, VGG, transformer, video_decoder, flow_net
from utils import sequence_decoder


class VSADisNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.video_cnn = video_cnn.ThreeDimCNN(out_feature=True)
        self.seq2seq = transformer.Transformer(consts.EMBED_SIZE, num_layers=3, dim_feedforward=1024)
        vocab_size = len(consts.VOCAB)
        self.embed = transformer.TokenEmbedding(vocab_size, consts.EMBED_SIZE)

        self.spatial_net = VGG.vgg13_bn()
        self.static_id_classifier = nn.Linear(consts.EMBED_SIZE, len(consts.VSA_TRAIN_USERS))

        self.flow_net = flow_net.FlowNet()
        self.motion_id_classifier = nn.Linear(consts.EMBED_SIZE, len(consts.VSA_TRAIN_USERS))

        self.video_decoder = video_decoder.VideoDecoder2(norm_mode="adin")

    def forward(self, inputs):
        """
        :param inputs: [original, same_id, same_content]
               -> element: [img_sequence, token, user_id, flow_sequence]
        :return:
        """
        bs = inputs[0][0].shape[0]
        features = self.encoder(inputs)
        concat_tokens = torch.cat((inputs[0][1], inputs[1][1], inputs[2][1]), dim=0)
        content_probs = self.content_recognizer(features[0]["pooled_feature"], concat_tokens)
        static_id_probs = self.static_id_classifier(features[1]["pooled_feature"])
        motion_id_probs = self.motion_id_classifier(features[2])

        features_for_decode = []
        probs = {"content_prob_list": content_probs.split(bs, dim=0),
                 "static_id_prob_list": static_id_probs.split(bs * consts.SAMPLE_IMG_NUM, dim=0),
                 "motion_id_prob_list": motion_id_probs.split(bs, dim=0),
                 "content_feature": features[0]["pooled_feature"].split(bs, dim=0),
                 "static_feature": features[1]["pooled_feature"].split(bs * consts.SAMPLE_IMG_NUM, dim=0),
                 "motion_feature": features[2].split(bs, dim=0),
                 }
        for i in range(3):
            features_for_decode.append({"content": features[0]["mid_feature"][bs * i: bs * (i + 1), ...],
                                        "static": features[1]["mid_feature"][
                                                  bs * consts.SAMPLE_IMG_NUM * i: bs * consts.SAMPLE_IMG_NUM * (i + 1),
                                                  ...]
                                       .reshape(bs, consts.SAMPLE_IMG_NUM, consts.EMBED_SIZE, 3, 6).mean(1),
                                        "flow": features[2][bs * i: bs * (i + 1), ...]})

        videos = self.decoder(features_for_decode)
        return probs, videos

    def encoder(self, inputs):
        """
        Inputs: [original, same_id, same_content]
        Returns: [content_features, static_features, flow_features]
        """
        content_features = self.video_cnn(torch.cat((inputs[0][0], inputs[1][0], inputs[2][0]), dim=0))
        pooled_feature_list = []
        mid_feature_list = []
        for i in range(3):
            img_sequence = inputs[i][0]
            b, c, l, h, w = img_sequence.shape
            img_sequence = img_sequence.permute(0, 2, 1, 3, 4).reshape(b * consts.SAMPLE_IMG_NUM, c, h, w)
            concat_static_features = self.spatial_net(img_sequence)
            pooled_feature_list.append(concat_static_features["pooled_feature"])
            mid_feature_list.append(concat_static_features["mid_feature"])
        static_id_features = {"pooled_feature": torch.cat(pooled_feature_list, dim=0),
                              "mid_feature": torch.cat(mid_feature_list, dim=0)}

        flow_features = self.flow_net(torch.cat((inputs[0][3], inputs[1][3], inputs[2][3])))
        return content_features, static_id_features, flow_features

    def decoder(self, features):
        content_mid_features = []
        static_mid_features = []
        flow_features = []
        for feature in features:
            content_mid_features.append(feature["content"])
            static_mid_features.append(feature["static"])
            flow_features.append(feature["flow"])
        id_a_content_a = [self.decode_one_video(content_mid_features[0], static_mid_features[0], flow_features[0]),
                          self.decode_one_video(content_mid_features[2], static_mid_features[0], flow_features[0]),
                          self.decode_one_video(content_mid_features[0], static_mid_features[0], flow_features[1]),
                          self.decode_one_video(content_mid_features[0], static_mid_features[1], flow_features[0])
                          ]

        id_a_content_b = [self.decode_one_video(content_mid_features[1], static_mid_features[1], flow_features[1]),
                          self.decode_one_video(content_mid_features[1], static_mid_features[1], flow_features[0]),
                          self.decode_one_video(content_mid_features[1], static_mid_features[0], flow_features[1])
                          ]

        id_b_content_a = [self.decode_one_video(content_mid_features[2], static_mid_features[2], flow_features[2]),
                          self.decode_one_video(content_mid_features[0], static_mid_features[2], flow_features[2])
                          ]

        return {"id_a_content_a": id_a_content_a, "id_a_content_b": id_a_content_b, "id_b_content_a": id_b_content_a}

    def decode_one_video(self, content, static, motion):
        static = static.unsqueeze(2)
        content = content + static  # not inplace
        video = self.video_decoder(content, latent=motion)
        return video

    def content_recognizer(self, content_feature, tokens):
        tokens = tokens[:, :-1]
        token_embed = self.embed(tokens)
        attn_mask = sequence_decoder.generate_square_subsequent_mask(tokens.shape[1])
        padding_mask = (tokens == consts.PAD)
        prob = self.seq2seq(content_feature, token_embed, src_mask=None, tgt_mask=attn_mask,
                            src_padding_mask=None, tgt_padding_mask=padding_mask,
                            memory_key_padding_mask=None, batch_first=True)
        prob = torch.matmul(prob, torch.transpose(self.embed.weight, 1, 0))
        return prob

    def predict(self, image_sequence):
        lip_feature = self.video_cnn(image_sequence)["pooled_feature"]
        lip_feature = lip_feature.permute(1, 0, 2)  # (frames, batch_size, embed_size)
        memory = self.seq2seq.encode(lip_feature.cuda(), src_mask=None)
        predict = sequence_decoder.beam_decode(self.seq2seq, self.embed, memory, consts.TOKEN_MAX_LEN)
        return predict
