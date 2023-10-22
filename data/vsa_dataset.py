import torch
import numpy as np
from torch.utils.data import Dataset
import os
import random
import cv2
import string

from data import consts, vocabulary


class VSADisentangle(Dataset):
    def __init__(self, mode, shuffle=True):
        if mode not in ("train", "test"):
            raise Exception("invalid dataset mode")
        self.mode = mode
        self.data_root = consts.DATA_ROOT + "/vsa"
        self.flow_root = consts.DATA_ROOT + "/vsa_flow"
        if mode == "train":
            users = consts.VSA_TRAIN_USERS
            group = [50, 35, 1]
        else:
            users = consts.VSA_TEST_USERS
            group = [20, 2, 20]
        user_list = ["user" + str(i) for i in users]

        self.user_list = user_list

        file_list = []
        random_appear_dict = {}
        fixed_appear_dict = {}
        for user in self.user_list:
            fixed_utterances = os.listdir(os.path.join(self.data_root, user, "fixed"))
            samples_0 = group[0] if len(fixed_utterances) >= group[0] else len(fixed_utterances)
            fixed_utterances = random.sample(fixed_utterances, samples_0)
            for fixed_utt in fixed_utterances:
                other_users = list(filter(lambda x: x != user, self.user_list))
                samples_1 = group[1] if len(other_users) >= group[1] else len(other_users)
                other_users = random.sample(other_users, samples_1)
                for other_user in other_users:
                    if fixed_utt not in os.listdir(os.path.join(self.data_root, other_user, "fixed")):
                        continue
                    random_utterances = os.listdir(os.path.join(self.data_root, user, "random"))
                    samples_2 = group[2] if len(random_utterances) >= group[2] else len(random_utterances)
                    random_utterances = random.sample(random_utterances, samples_2)
                    for random_utt in random_utterances:
                        random_path = "/".join([user, "random", random_utt])
                        fixed_path = "/".join([user, "fixed", fixed_utt])
                        item = [fixed_path, random_path, "/".join([other_user, "fixed", fixed_utt])]
                        if random_path not in random_appear_dict:
                            random_appear_dict[random_path] = [item]
                        else:
                            random_appear_dict[random_path].append(item)
                        if fixed_path not in fixed_appear_dict:
                            fixed_appear_dict[fixed_path] = [item]
                        else:
                            fixed_appear_dict[fixed_path].append(item)
        for k, v in random_appear_dict.items():
            sample_num = 1 if len(v) >= 1 else len(v)
            file_list.extend(random.sample(v, sample_num))
        for k, v in fixed_appear_dict.items():
            sample_num = 1 if len(v) >= 1 else len(v)
            file_list.extend(random.sample(v, sample_num))

        if shuffle:
            random.shuffle(file_list)
        self.file_list = file_list

    def get_images_and_token(self, img_path):
        s = img_path.split("/")
        horizon_flip = self.mode == "train" and random.random() > 0.5
        img_sequence_list = []
        for index, img_name in enumerate(sorted(os.listdir(img_path))):
            img = cv2.imread(os.path.join(img_path, img_name))  # (H, W, c)
            img = img / 255.0
            img_tensor = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
            img_sequence_list.append(img_tensor)
        img_sequence = torch.stack(img_sequence_list[:50], 1)
        flow_data = np.load(os.path.join(self.flow_root, s[-3], s[-2], s[-1] + ".npy"))
        flow_data = flow_data[:49, :, :, :]
        flow_data = np.transpose(flow_data, (1, 0, 2, 3))
        flow_sequence = torch.from_numpy(flow_data)

        if horizon_flip:
            img_sequence = torch.flip(img_sequence, dims=[3])
            flow_sequence = torch.flip(flow_sequence, dims=[3])

        label = s[-1]
        user_id = self.user_list.index(s[-3])
        full = " ".join([consts.NUMBER[n] for n in label])
        token = vocabulary.txt2token(full, consts.VOCAB)
        token.insert(0, consts.BOS)
        token.append(consts.EOS)
        for i in range(consts.TOKEN_MAX_LEN - len(token)):
            token.append(consts.PAD)

        token = torch.tensor(token, dtype=torch.int64)
        return img_sequence, token, user_id, flow_sequence

    def __getitem__(self, index):
        img_files = self.file_list[index]
        return [self.get_images_and_token(os.path.join(self.data_root, f)) for f in img_files]

    def __len__(self):
        return len(self.file_list)
