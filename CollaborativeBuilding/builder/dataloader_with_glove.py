from torch.functional import Tensor
from torch.utils.data import Dataset, DataLoader
import torch
import argparse
import tqdm
import numpy as np
import pickle
import os, sys
sys.path.append('..')
from utils import type2id, load_pkl_data, x_min, x_max, y_min, y_max, z_min, z_max, BuilderAction, f2, stop_action_label
from builder.data_loader_with_glove import BuilderActionExample, Region
from builder.vocab import Vocabulary
from builder.utils_builder import get_feasibile_location

color2id = {
	"orange": 0,
	"red": 1,
	"green": 2,
	"blue": 3,
	"purple": 4,
	"yellow": 5
}

id2color = {v: k for k, v in type2id.items()}

action2id = {
	"placement": 0,
	"removal": 1
}

class BuilderDataset(Dataset):
    def __init__(self, args, split, encoder_vocab):
        if args.load_items:
            self.dataset_dir = args.json_data_dir
            self.split = split
            self.samples = False
            with open(os.path.join(self.dataset_dir, self.split + "-items.pkl"), 'rb') as f:
                self.items = torch.load(f, map_location="cpu") 
            self.include_empty_channel = True
            self.use_builder_actions = False
            self.add_action_history_weight = True
            self.action_history_weighting_scheme = "step"
            self.concatenate_action_history_weight = True
            self.add_perspective_coords = False
        else:
            self.include_empty_channel = args.include_empty_channel
            self.use_builder_actions = args.use_builder_actions
            self.add_action_history_weight = args.add_action_history_weight
            self.action_history_weighting_scheme = args.action_history_weighting_scheme
            self.concatenate_action_history_weight = args.concatenate_action_history_weight
            self.num_prev_utterances = args.num_prev_utterances
            self.use_builder_actions = args.use_builder_actions
            self.add_perspective_coords = args.add_perspective_coords
            self.encoder_vocab = encoder_vocab

            self.split = split
            self.max_length = args.max_length
            self.dataset_dir = args.json_data_dir
            self.samples = load_pkl_data(self.dataset_dir + "/"+ self.split + "-samples.pkl")
            self.items = []
            for i in tqdm.tqdm(range(len(self))):
                self.items.append(self.preprocess(i))
            torch.save(self.items, os.path.join(self.dataset_dir, self.split + "-items.pkl")) 
            print('{} data has been stored at {}'.format(self.split, os.path.join(self.dataset_dir, self.split + "-items.pkl")))

    def __len__(self):
        if self.samples:
            return len(self.samples)
        else:
            return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

    def preprocess(self, idx):
        """ Computes the tensor representations of a sample """
        orig_sample = self.samples[idx]

        all_actions = orig_sample["next_builder_actions"]
        perspective_coords = orig_sample["perspective_coordinates"]

        initial_prev_config_raw = all_actions[0].prev_config
        initial_action_history_raw = all_actions[0].action_history
        end_built_config_raw = all_actions[-1].built_config

        all_actions_reprs = list(map(lambda x: self.get_repr(x, perspective_coords), all_actions))
        all_grid_repr_inputs = list(map(lambda x: x[0], all_actions_reprs))
        all_outputs = list(map(lambda x: x[1], all_actions_reprs))
        all_location_mask = list(map(lambda x: x[2], all_actions_reprs))

        all_actions_repr_inputs = list(map(f2, all_actions))

        start_action_repr = torch.Tensor([0] * 11)
        
        stop_action = BuilderActionExample(
            action = None,
            built_config = all_actions[-1].built_config,
            prev_config = all_actions[-1].built_config,
            action_history = all_actions[-1].action_history + [all_actions[-1].action]
        )
        stop_action_repr = self.get_repr(stop_action, perspective_coords)
        stop_action_grid_repr_input = stop_action_repr[0]
        stop_action_output_label = stop_action_repr[1]
        stop_action_location_mask = stop_action_repr[2]

        dec_inputs_1 = all_grid_repr_inputs + [stop_action_grid_repr_input]
        dec_inputs_2 = [start_action_repr] + all_actions_repr_inputs
        location_mask = all_location_mask + [stop_action_location_mask]

        dec_outputs = all_outputs + [stop_action_output_label]
        
        # Encoder inputs
        utterances_to_add = []

        '''
        an example of orig_sample["prev_utterances"]:
        
        [{'speaker': 'Builder', 'utterance': ['<dialogue>']}, 
        {'speaker': 'Builder', 'utterance': ['Mission has started.']}, 
        {'speaker': 'Builder', 'utterance': ['hi']}, 
        {'speaker': 'Architect', 'utterance': ["hello, it looks like we are building a cube where the sides are made of the letters 'a', 'b', 'c', 'd'"]}, 
        {'speaker': 'Builder', 'utterance': ['sounds cool']}, 
        {'speaker': 'Architect', 'utterance': ['First, we will need to make a red A, make a 3 block long red bar somewhere near the middle']}]
        '''
 
        i = 0
        utterances_idx = len(orig_sample["prev_utterances"])-1
        while i < self.num_prev_utterances: 
            if utterances_idx < 0:
                break

            prev = orig_sample["prev_utterances"][utterances_idx]
            speaker = prev["speaker"]
            utterance = prev["utterance"]

            if "<builder_" in utterance[0]:
                if self.use_builder_actions:
                    utterances_to_add.insert(0, prev)
                i -= 1
            elif "mission has started ." in " ".join(utterance) and 'Builder' in speaker:
                i -= 1 

            else:
                utterances_to_add.insert(0, prev)

            utterances_idx -= 1
            i += 1

        prev_utterances = []

        for prev in utterances_to_add:
            speaker = prev["speaker"]
            utterance = prev["utterance"]

            if "<dialogue>" in utterance[0]:
                prev_utterances.append(self.encoder_vocab('<dialogue>'))

            elif "<builder_" in utterance[0]:
                if self.use_builder_actions:
                    prev_utterances.append(self.encoder_vocab(utterance[0]))

            else:
                start_token = self.encoder_vocab('<architect>') if 'Architect' in speaker else self.encoder_vocab('<builder>')
                end_token = self.encoder_vocab('</architect>') if 'Architect' in speaker else self.encoder_vocab('</builder>')
                prev_utterances.append(start_token)
                prev_utterances.extend(self.encoder_vocab(token) for token in utterance)
                prev_utterances.append(end_token)  
        
        if len(prev_utterances) > self.max_length:
            prev_utterances = prev_utterances[-100:]

        while len(prev_utterances) < self.max_length:
            prev_utterances.append(self.encoder_vocab.word2idx['<pad>'])

        assert len(prev_utterances) == 100
        
        return (
            torch.Tensor(prev_utterances),
            torch.stack(dec_inputs_1), ## torch.Size([action, 8, 11, 9, 11])
            torch.stack(dec_inputs_2), ## torch.Size([action, 11])
            torch.Tensor(dec_outputs), ## torch.Size([action])
            torch.Tensor(location_mask), ## torch.Size([action, 1089])
            RawInputs(initial_prev_config_raw, initial_action_history_raw, end_built_config_raw, perspective_coords)
        )

    def get_repr(self, builder_action, perspective_coords):
        config = builder_action.prev_config
        for block in config:
            for key in ['x', 'y', 'z']:
                block[key] = int(block[key])

        cell_samples = split_orig_sample(builder_action)

        cell_repr_size = 6
        if self.include_empty_channel:
            cell_repr_size += 1
        if self.add_action_history_weight and self.concatenate_action_history_weight:
            cell_repr_size += 1

        cell_samples_reprs_4d = torch.zeros([cell_repr_size, 11, 9, 11]) 
        cell_samples_labels = []

        for sample in cell_samples:
            cell = Region(
                x_min = sample["x"], x_max = sample["x"], y_min = sample["y"], y_max = sample["y"], z_min = sample["z"], z_max = sample["z"]
            )

            def get_current_state(cell, built_config):
                current_state = "empty"

                for block in built_config:
                    if cell.x_min == block["x"] and cell.y_min == block["y"] and cell.z_min == block["z"]:
                        current_state = block["type"]
                        break

                return current_state
            current_state = get_current_state(cell, sample["built_config"]) 

            def get_action_history_weight(cell, action_history):
                if self.add_action_history_weight:
                    action_history_weight = 0.0

                    for i in range(len(action_history) - 1, -1, -1):
                        action = action_history[i]
                        if cell.x_min == action.block["x"] and cell.y_min == action.block["y"] and cell.z_min == action.block["z"]:
                            if self.action_history_weighting_scheme == "smooth":
                                action_history_weight = (i + 1)/len(action_history)
                            elif self.action_history_weighting_scheme == "step":
                                action_history_weight = i - (len(action_history) - 1) + 5
                                action_history_weight = np.maximum(action_history_weight, 0.0)
                            break
                else:
                    action_history_weight = None

                return action_history_weight
            action_history_weight = get_action_history_weight(cell, sample["builder_action_history"])

            # output label
            if sample["action_type"] == "placement":
                output_label = type2id[sample["block_type"]]
            elif sample["action_type"] == "removal":
                output_label = len(type2id)
            else: 
                output_label = len(type2id) + 1

            # convert data to tensors
            def f(current_cell_state):
                if current_cell_state in type2id:
                    index = type2id[current_cell_state]
                else:
                    if self.include_empty_channel:
                        index = len(type2id) # 6
                    else:
                        index = None

                if self.include_empty_channel:
                    vec_length = len(type2id) + 1
                else:
                    vec_length = len(type2id)

                one_hot_vec = [0] * vec_length
                if index != None:
                    one_hot_vec[index] = 1

                return one_hot_vec

            current_cell_state_one_hot_vec = f(current_state)

            action_history_weight_vec = [action_history_weight]

            def get_joint_repr(vec1, vec2):
                if not vec2 == [None]:
                    # get joint repr for a region
                    if self.concatenate_action_history_weight:
                        # concatenate
                        new_vec = vec1 + vec2
                    else:
                        # find the 1.0 in vec1 if any and replace it by vec2[0] if latter is >= 1.0
                        new_vec = list(map(
                            lambda x: x if x == 0 or (x == 1 and vec2[0] == 0) else vec2[0],
                            vec1
                        ))

                    return new_vec
                else:
                    return vec1
            repr_vec = get_joint_repr(current_cell_state_one_hot_vec, action_history_weight_vec) ## list

            cell_samples_reprs_4d[:, sample["x"]+5, sample["y"]-1, sample["z"]+5] = torch.Tensor(repr_vec) # offsets needed to map to range [0, something]
            cell_samples_labels.append(output_label)
        
        location_mask = get_feasibile_location(sample["built_config"])

        action_cell_index = None
        action_cell_label = None
        for i, label in enumerate(cell_samples_labels):
            if label < 7:
                action_cell_index 