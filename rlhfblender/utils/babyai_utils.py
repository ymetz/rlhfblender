import json
import os
import re
from abc import ABC, abstractmethod

import numpy
import torch


class DictList(dict):
    """A dictionnary of lists of same size. Dictionnary items can be
    accessed using `.` notation and list items using `[]` notation.

    Example:
        >>> d = DictList({"a": [[1, 2], [3, 4]], "b": [[5], [6]]})
        >>> d.a
        [[1, 2], [3, 4]]
        >>> d[0]
        DictList({"a": [1, 2], "b": [5]})
    """

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __len__(self):
        return len(next(iter(dict.values(self))))

    def __getitem__(self, index):
        return DictList({key: value[index] for key, value in dict.items(self)})

    def __setitem__(self, index, d):
        for key, value in d.items():
            dict.__getitem__(self, key)[index] = value

def create_folders_if_necessary(path):
    dirname = os.path.dirname(path)
    if not (os.path.isdir(dirname)):
        os.makedirs(dirname)

def storage_dir():
    # defines the storage directory to be in the root (Same level as babyai folder)
    return os.environ.get("BABYAI_STORAGE", ".")

def get_model_dir(model_name):
    return os.path.join(storage_dir(), "models", model_name)

def get_model_path(model_name):
    return os.path.join(get_model_dir(model_name), "model.pt")

def load_model(model_name, raise_not_found=True):
    path = get_model_path(model_name)
    try:
        if torch.cuda.is_available():
            model = torch.load(path)
        else:
            model = torch.load(path, map_location=torch.device("cpu"))
        model.eval()
        return model
    except FileNotFoundError:
        if raise_not_found:
            raise FileNotFoundError("No model found at {}".format(path))

def get_vocab_path(model_name):
    return os.path.join(get_model_dir(model_name), "vocab.json")


class Vocabulary:
    def __init__(self, model_name):
        self.path = get_vocab_path(model_name)
        self.max_size = 100
        if os.path.exists(self.path):
            self.vocab = json.load(open(self.path))
        else:
            self.vocab = {}

    def __getitem__(self, token):
        if token not in self.vocab.keys():
            if len(self.vocab) >= self.max_size:
                raise ValueError("Maximum vocabulary capacity reached")
            self.vocab[token] = len(self.vocab) + 1
        return self.vocab[token]

    def save(self, path=None):
        if path is None:
            path = self.path
        create_folders_if_necessary(path)
        json.dump(self.vocab, open(path, "w"))

    def copy_vocab_from(self, other):
        """
        Copy the vocabulary of another Vocabulary object to the current object.
        """
        self.vocab.update(other.vocab)


class InstructionsPreprocessor(object):
    def __init__(self, model_name, load_vocab_from=None):
        self.model_name = model_name
        self.vocab = Vocabulary(model_name)

        path = get_vocab_path(model_name)
        if not os.path.exists(path) and load_vocab_from is not None:
            # self.vocab.vocab should be an empty dict
            secondary_path = os.path.join(load_vocab_from, "vocab.json")
            if os.path.exists(secondary_path):
                old_vocab = Vocabulary(load_vocab_from)
                self.vocab.copy_vocab_from(old_vocab)
            else:
                raise FileNotFoundError("No pre-trained model under the specified name")

    def __call__(self, obss, device=None):
        raw_instrs = []
        max_instr_len = 0

        for obs in obss:
            tokens = re.findall("([a-z]+)", obs["mission"].lower())
            instr = numpy.array([self.vocab[token] for token in tokens])
            raw_instrs.append(instr)
            max_instr_len = max(len(instr), max_instr_len)

        instrs = numpy.zeros((len(obss), max_instr_len))

        for i, instr in enumerate(raw_instrs):
            instrs[i, : len(instr)] = instr

        instrs = torch.tensor(instrs, device=device, dtype=torch.long)
        return instrs


class RawImagePreprocessor(object):
    def __call__(self, obss, device=None):
        images = numpy.array([obs["image"] for obs in obss])
        images = torch.tensor(images, device=device, dtype=torch.float)
        return images


class IntImagePreprocessor(object):
    def __init__(self, num_channels, max_high=255):
        self.num_channels = num_channels
        self.max_high = max_high
        self.offsets = numpy.arange(num_channels) * max_high
        self.max_size = int(num_channels * max_high)

    def __call__(self, obss, device=None):
        images = numpy.array([obs["image"] for obs in obss])
        # The padding index is 0 for all the channels
        images = (images + self.offsets) * (images > 0)
        images = torch.tensor(images, device=device, dtype=torch.long)
        return images


class ObssPreprocessor:
    def __init__(self, model_name, obs_space=None, load_vocab_from=None):
        self.image_preproc = RawImagePreprocessor()
        self.instr_preproc = InstructionsPreprocessor(model_name, load_vocab_from)
        self.vocab = self.instr_preproc.vocab
        self.obs_space = {"image": 147, "instr": self.vocab.max_size}

    def __call__(self, obss, device=None):
        obs_ = DictList()

        if "image" in self.obs_space.keys():
            obs_.image = self.image_preproc(obss, device=device)

        if "instr" in self.obs_space.keys():
            obs_.instr = self.instr_preproc(obss, device=device)

        return obs_


class IntObssPreprocessor(object):
    def __init__(self, model_name, obs_space, load_vocab_from=None):
        image_obs_space = obs_space.spaces["image"]
        self.image_preproc = IntImagePreprocessor(
            image_obs_space.shape[-1], max_high=image_obs_space.high.max()
        )
        self.instr_preproc = InstructionsPreprocessor(load_vocab_from or model_name)
        self.vocab = self.instr_preproc.vocab
        self.obs_space = {
            "image": self.image_preproc.max_size,
            "instr": self.vocab.max_size,
        }

    def __call__(self, obss, device=None):
        obs_ = DictList()

        if "image" in self.obs_space.keys():
            obs_.image = self.image_preproc(obss, device=device)

        if "instr" in self.obs_space.keys():
            obs_.instr = self.instr_preproc(obss, device=device)

        return obs_

class Agent(ABC):
    """An abstraction of the behavior of an agent. The agent is able:
    - to choose an action given an observation,
    - to analyze the feedback (i.e. reward and done state) of its action."""

    def on_reset(self):
        pass

    @abstractmethod
    def act(self, obs):
        """Propose an action based on observation.

        Returns a dict, with 'action` entry containing the proposed action,
        and optionaly other entries containing auxiliary information
        (e.g. value function).

        """

    @abstractmethod
    def analyze_feedback(self, reward, done):
        pass


class ModelAgent(Agent):
    """A model-based agent. This agent behaves using a model."""

    def __init__(self, model_or_name, obss_preprocessor, argmax):
        if obss_preprocessor is None:
            assert isinstance(model_or_name, str)
            obss_preprocessor = ObssPreprocessor(model_or_name)
        self.obss_preprocessor = obss_preprocessor
        if isinstance(model_or_name, str):
            self.model = load_model(model_or_name)
            if torch.cuda.is_available():
                self.model.cuda()
        else:
            self.model = model_or_name
        self.device = next(self.model.parameters()).device
        self.argmax = argmax
        self.memory = None

    def act_batch(self, many_obs):
        if self.memory is None:
            self.memory = torch.zeros(
                len(many_obs), self.model.memory_size, device=self.device
            )
        elif self.memory.shape[0] != len(many_obs):
            raise ValueError("stick to one batch size for the lifetime of an agent")
        preprocessed_obs = self.obss_preprocessor(many_obs, device=self.device)

        with torch.no_grad():
            model_results = self.model(preprocessed_obs, self.memory)
            dist = model_results["dist"]
            value = model_results["value"]
            self.memory = model_results["memory"]

        if self.argmax:
            action = dist.probs.argmax(1)
        else:
            action = dist.sample()

        return {"action": action, "dist": dist, "value": value}

    def act(self, obs):
        return self.act_batch([obs])

    def analyze_feedback(self, reward, done):
        if isinstance(done, tuple):
            for i in range(len(done)):
                if done[i]:
                    self.memory[i, :] *= 0.0
        else:
            self.memory *= 1 - done