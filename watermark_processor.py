# coding=utf-8
# Copyright 2023 Authors of "A Watermark for Large Language Models"
# available at https://arxiv.org/abs/2301.10226
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
import collections
from math import sqrt
from itertools import chain, tee
from functools import lru_cache
import numpy as np

import scipy.stats
import torch
from tokenizers import Tokenizer
from transformers import LogitsProcessor

from normalizers import normalization_strategy_lookup
from alternative_prf_schemes import prf_lookup, seeding_scheme_lookup
import torch.nn.functional as F
from alternative_prf_schemes import hashint

class WatermarkBase:
    def __init__(
        self,
        vocab: list[int] = None,
        gamma: float = 0.5,
        delta: float = 2.0,
        seeding_scheme: str = "simple_1",  # simple default, find more schemes in alternative_prf_schemes.py
        select_green_tokens: bool = True,  # should always be the default if not running in legacy mode
    ):
        # patch now that None could now maybe be passed as seeding_scheme
        if seeding_scheme is None:
            seeding_scheme = "simple_1"

        # Vocabulary setup
        self.vocab = vocab
        self.vocab_size = len(vocab)

        # Watermark behavior:
        self.gamma = gamma
        self.delta = delta
        self.rng = None
        self._initialize_seeding_scheme(seeding_scheme)
        # Legacy behavior:
        self.select_green_tokens = select_green_tokens

    def _initialize_seeding_scheme(self, seeding_scheme: str) -> None:
        """Initialize all internal settings of the seeding strategy from a colloquial, "public" name for the scheme."""
        self.prf_type, self.context_width, self.self_salt, self.hash_key = seeding_scheme_lookup(
            seeding_scheme
        )

    def _seed_rng(self, input_ids: torch.LongTensor) -> None:
        """Seed RNG from local context. Not batched, because the generators we use (like cuda.random) are not batched."""
        # Need to have enough context for seed generation
        if input_ids.shape[-1] < self.context_width:
            raise ValueError(
                f"seeding_scheme requires at least a {self.context_width} token prefix to seed the RNG."
            )

        prf_key = prf_lookup[self.prf_type](
            input_ids[-self.context_width :], salt_key=self.hash_key
        )
        # enable for long, interesting streams of pseudorandom numbers: print(prf_key)
        self.rng.manual_seed(prf_key % (2**64 - 1))  # safeguard against overflow from long

    def _get_greenlist_ids(self, input_ids: torch.LongTensor) -> torch.LongTensor:
        """Seed rng based on local context width and use this information to generate ids on the green list."""
        self._seed_rng(input_ids)

        greenlist_size = int(self.vocab_size * self.gamma)
        vocab_permutation = torch.randperm(
            self.vocab_size, device=input_ids.device, generator=self.rng
        )
        if self.select_green_tokens:  # directly
            greenlist_ids = vocab_permutation[:greenlist_size]  # new
        else:  # select green via red
            greenlist_ids = vocab_permutation[
                (self.vocab_size - greenlist_size) :
            ]  # legacy behavior
        # print(greenlist_size)
        # print(len(greenlist_ids))
        # import pdb; pdb.set_trace()
        # input("check")
        return greenlist_ids


class WatermarkLogitsProcessor(WatermarkBase, LogitsProcessor):
    """LogitsProcessor modifying model output scores in a pipe. Can be used in any HF pipeline to modify scores to fit the watermark,
    but can also be used as a standalone tool inserted for any model producing scores inbetween model outputs and next token sampler.
    """

    def __init__(self, *args, store_spike_ents: bool = False, **kwargs):
        super().__init__(*args, **kwargs)

        self.store_spike_ents = store_spike_ents
        self.spike_entropies = None
        if self.store_spike_ents:
            self._init_spike_entropies()

    def _init_spike_entropies(self):
        alpha = torch.exp(torch.tensor(self.delta)).item()
        gamma = self.gamma

        self.z_value = ((1 - gamma) * (alpha - 1)) / (1 - gamma + (alpha * gamma))
        self.expected_gl_coef = (gamma * alpha) / (1 - gamma + (alpha * gamma))

        # catch for overflow when bias is "infinite"
        if alpha == torch.inf:
            self.z_value = 1.0
            self.expected_gl_coef = 1.0

    def _get_spike_entropies(self):
        spike_ents = [[] for _ in range(len(self.spike_entropies))]
        for b_idx, ent_tensor_list in enumerate(self.spike_entropies):
            for ent_tensor in ent_tensor_list:
                spike_ents[b_idx].append(ent_tensor.item())
        return spike_ents

    def _get_and_clear_stored_spike_ents(self):
        spike_ents = self._get_spike_entropies()
        self.spike_entropies = None
        return spike_ents

    def _compute_spike_entropy(self, scores):
        # precomputed z value in init
        probs = scores.softmax(dim=-1)
        denoms = 1 + (self.z_value * probs)
        renormed_probs = probs / denoms
        sum_renormed_probs = renormed_probs.sum()
        return sum_renormed_probs

    def _calc_greenlist_mask(
        self, scores: torch.FloatTensor, greenlist_token_ids
    ) -> torch.BoolTensor:
        # Cannot lose loop, greenlists might have different lengths
        green_tokens_mask = torch.zeros_like(scores, dtype=torch.bool)
        for b_idx, greenlist in enumerate(greenlist_token_ids):
            if len(greenlist) > 0:
                green_tokens_mask[b_idx][greenlist] = True
        return green_tokens_mask

    def _bias_greenlist_logits(
        self, scores: torch.Tensor, greenlist_mask: torch.Tensor, greenlist_bias: float
    ) -> torch.Tensor:
        scores[greenlist_mask] = scores[greenlist_mask] + greenlist_bias
        return scores

    def _score_rejection_sampling(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, tail_rule="fixed_compute"
    ) -> list[int]:
        """Generate greenlist based on current candidate next token. Reject and move on if necessary. Method not batched.
        This is only a partial version of Alg.3 "Robust Private Watermarking", as it always assumes greedy sampling. It will still (kinda)
        work for all types of sampling, but less effectively.
        To work efficiently, this function can switch between a number of rules for handling the distribution tail.
        These are not exposed by default.
        """
        sorted_scores, greedy_predictions = scores.sort(dim=-1, descending=True)

        final_greenlist = []
        for idx, prediction_candidate in enumerate(greedy_predictions):
            greenlist_ids = self._get_greenlist_ids(
                torch.cat([input_ids, prediction_candidate[None]], dim=0)
            )  # add candidate to prefix
            if prediction_candidate in greenlist_ids:  # test for consistency
                final_greenlist.append(prediction_candidate)
            # What follows below are optional early-stopping rules for efficiency
            if tail_rule == "fixed_score":
                if sorted_scores[0] - sorted_scores[idx + 1] > self.delta:
                    break
            elif tail_rule == "fixed_list_length":
                if len(final_greenlist) == 10:
                    break
            elif tail_rule == "fixed_compute":
                if idx == 40:
                    break
            else:
                pass  # do not break early
        return torch.as_tensor(final_greenlist, device=input_ids.device)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Call with previous context as input_ids, and scores for next token."""

        # this is lazy to allow us to co-locate on the watermarked model's device
        self.rng = torch.Generator(device=input_ids.device) if self.rng is None else self.rng

        # NOTE, it would be nice to get rid of this batch loop, but currently,
        # the seed and partition operations are not tensor/vectorized, thus
        # each sequence in the batch needs to be treated separately.

        # print(self.self_salt)
        # import pdb; pdb.set_trace()

        list_of_greenlist_ids = [None for _ in input_ids]  # Greenlists could differ in length
        for b_idx, input_seq in enumerate(input_ids):
            if self.self_salt:
                # input("greenlist_ids this one")
                greenlist_ids = self._score_rejection_sampling(input_seq, scores[b_idx])
            else:
                # input("check here")
                greenlist_ids = self._get_greenlist_ids(input_seq)
            #     print(len(greenlist_ids))
            # import pdb; pdb.set_trace()

            list_of_greenlist_ids[b_idx] = greenlist_ids

            # logic for computing and storing spike entropies for analysis
            if self.store_spike_ents:
                if self.spike_entropies is None:
                    self.spike_entropies = [[] for _ in range(input_ids.shape[0])]
                self.spike_entropies[b_idx].append(self._compute_spike_entropy(scores[b_idx]))

        green_tokens_mask = self._calc_greenlist_mask(
            scores=scores, greenlist_token_ids=list_of_greenlist_ids
        )
        # print(type(green_tokens_mask))
        # import pdb; pdb.set_trace()
        scores = self._bias_greenlist_logits(
            scores=scores, greenlist_mask=green_tokens_mask, greenlist_bias=self.delta
        )

        return scores


class SemWatermarkLogitsProcessor(WatermarkBase, LogitsProcessor):
    """LogitsProcessor modifying model output scores in a pipe. Can be used in any HF pipeline to modify scores to fit the watermark,
    but can also be used as a standalone tool inserted for any model producing scores inbetween model outputs and next token sampler.
    """

    def __init__(self, *args, cl_mlp, store_spike_ents: bool = False, mean_pooling, cl_pooling_method, cl_k, flag_kmeans, kmeans_label_file="kmean", discrete_value_number=10, **kwargs):
        super().__init__(*args, **kwargs)

        self.store_spike_ents = store_spike_ents
        self.spike_entropies = None
        self.cl_mlp = cl_mlp
        self.mean_pooling = mean_pooling
        self.k = cl_k
        self.discrete_value_number = discrete_value_number
        self.cl_pooling_method = cl_pooling_method
        self.flag_kmeans = flag_kmeans
        self.kmeans_green_cluster_lookup = None
        if self.flag_kmeans:
            self.kmeans_label = np.load(f"./results/{kmeans_label_file}.npy")

        if self.store_spike_ents:
            self._init_spike_entropies()

        self.greenlist_table = {}

    def _init_spike_entropies(self):
        alpha = torch.exp(torch.tensor(self.delta)).item()
        gamma = self.gamma

        self.z_value = ((1 - gamma) * (alpha - 1)) / (1 - gamma + (alpha * gamma))
        self.expected_gl_coef = (gamma * alpha) / (1 - gamma + (alpha * gamma))

        # catch for overflow when bias is "infinite"
        if alpha == torch.inf:
            self.z_value = 1.0
            self.expected_gl_coef = 1.0

    def _get_spike_entropies(self):
        spike_ents = [[] for _ in range(len(self.spike_entropies))]
        for b_idx, ent_tensor_list in enumerate(self.spike_entropies):
            for ent_tensor in ent_tensor_list:
                spike_ents[b_idx].append(ent_tensor.item())
        return spike_ents

    def _get_and_clear_stored_spike_ents(self):
        spike_ents = self._get_spike_entropies()
        self.spike_entropies = None
        return spike_ents

    def _compute_spike_entropy(self, scores):
        # precomputed z value in init
        probs = scores.softmax(dim=-1)
        denoms = 1 + (self.z_value * probs)
        renormed_probs = probs / denoms
        sum_renormed_probs = renormed_probs.sum()
        return sum_renormed_probs

    def _calc_greenlist_mask(
        self, scores: torch.FloatTensor, greenlist_token_ids
    ) -> torch.BoolTensor:
        # Cannot lose loop, greenlists might have different lengths
        green_tokens_mask = torch.zeros_like(scores, dtype=torch.bool)
        for b_idx, greenlist in enumerate(greenlist_token_ids):
            if len(greenlist) > 0:
                green_tokens_mask[b_idx][greenlist] = True
        return green_tokens_mask

    def _bias_greenlist_logits(
        self, scores: torch.Tensor, greenlist_mask: torch.Tensor, greenlist_bias: float
    ) -> torch.Tensor:
        scores[greenlist_mask] = scores[greenlist_mask] + greenlist_bias
        return scores

    def _sem_seed_rng(self, hidden_embeddings: torch.LongTensor, cl_mlp) -> None:
        """Seed RNG from local context. Not batched, because the generators we use (like cuda.random) are not batched."""
        if "sem" in self.prf_type:
            prf_key, a = prf_lookup[self.prf_type](
                    hidden_embeddings=F.normalize(hidden_embeddings, p=2, dim=1), salt_key=self.hash_key, cl_mlp=cl_mlp, discrete_value_number=self.discrete_value_number
                )
        else:
            prf_key = prf_lookup[self.prf_type](
                    hidden_embeddings=F.normalize(hidden_embeddings, p=2, dim=1), salt_key=self.hash_key, cl_mlp=cl_mlp
                )
        self.rng.manual_seed(prf_key % (2**64 - 1))  # safeguard against overflow from long

        return a

    def _sem_get_greenlist_ids(self, hidden_embeddings: torch.LongTensor, cl_mlp) -> torch.LongTensor:
        """Seed rng based on local context width and use this information to generate ids on the green list."""
        a = self._sem_seed_rng(hidden_embeddings=hidden_embeddings, cl_mlp=cl_mlp)

        if self.flag_kmeans:
            greenlist_ids = self.kmeans_green_cluster_lookup[a].detach()
            
        else:
            greenlist_size = int(self.vocab_size * self.gamma)
            vocab_permutation = torch.randperm(
                self.vocab_size, device=hidden_embeddings.device, generator=self.rng
            )
            if self.select_green_tokens:  # directly
                greenlist_ids = vocab_permutation[:greenlist_size]  # new
            else:  # select green via red
                greenlist_ids = vocab_permutation[
                    (self.vocab_size - greenlist_size) :
                ]  # legacy behavior
        # self.greenlist_table[a.item()] = greenlist_ids#.detach().cpu().numpy()
        # if len(self.greenlist_table) == self.discrete_value_number:
        #     import pdb; pdb.set_trace()
        return greenlist_ids, a

    def _score_rejection_sampling(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, tail_rule="fixed_compute"
    ) -> list[int]:
        raise("Not emplmented.")
        # input("check _score_rejection_sampling")
        """Generate greenlist based on current candidate next token. Reject and move on if necessary. Method not batched.
        This is only a partial version of Alg.3 "Robust Private Watermarking", as it always assumes greedy sampling. It will still (kinda)
        work for all types of sampling, but less effectively.
        To work efficiently, this function can switch between a number of rules for handling the distribution tail.
        These are not exposed by default.
        """
        sorted_scores, greedy_predictions = scores.sort(dim=-1, descending=True)

        final_greenlist = []
        seed_list = None
        for idx, prediction_candidate in enumerate(greedy_predictions):
            greenlist_ids, prf_seed = self._sem_get_greenlist_ids(
                torch.cat([input_ids, prediction_candidate[None]], dim=0)
            )  # add candidate to prefix
            seed_list = prf_seed
            if prediction_candidate in greenlist_ids:  # test for consistency
                final_greenlist.append(prediction_candidate)

            # What follows below are optional early-stopping rules for efficiency
            # print(sorted_scores[0], sorted_scores[-1])
            # input("check here")
            if tail_rule == "fixed_score":
                if sorted_scores[0] - sorted_scores[idx + 1] > self.delta:
                    break
            elif tail_rule == "fixed_list_length":
                if len(final_greenlist) == 10:
                    break
            elif tail_rule == "fixed_compute":
                if idx == 40:
                    break
            else:
                pass  # do not break early
        raise("not sure whether it is right")
        return torch.as_tensor(final_greenlist, device=input_ids.device), seed_list

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, hidden_embeddings: torch.FloatTensor) -> torch.FloatTensor:
        """Call with previous context as input_ids, and scores for next token."""

        # this is lazy to allow us to co-locate on the watermarked model's device
        self.rng = torch.Generator(device=input_ids.device) if self.rng is None else self.rng
        if self.kmeans_green_cluster_lookup is None and self.flag_kmeans:
            # print("check")
            # input("check")
            yuliang = True
            if yuliang:
                self.kmeans_green_cluster_lookup = []
                self.cluster_num = np.max(self.kmeans_label)
                green_cluster_size = int(self.cluster_num * (self.gamma + 0.1))
                green_ids_size = int(self.vocab_size * self.gamma)
                for a in range(self.discrete_value_number+1):
                    # print(a)
                    prf_key = hashint(torch.tensor(self.hash_key * a)).item()
                    self.rng.manual_seed(prf_key % (2**64 - 1))
                    cluster_permutation = torch.randperm(
                        self.cluster_num, device=hidden_embeddings.device, generator=self.rng
                    )
                    # print("check 01")
                    if self.select_green_tokens:
                        green_cluster_ids = cluster_permutation[:green_cluster_size]
                    else:
                        green_cluster_ids = cluster_permutation[
                            (self.cluster_num - green_cluster_size) :
                        ]
                    greenlist_ids = []
                    print("check 02")
                    green_cluster_ids_npy = green_cluster_ids.cpu().detach().numpy()
                    # for _id in range(self.kmeans_label.shape[0]):
                    #     if self.kmeans_label[_id] in green_cluster_ids_npy:
                    #         greenlist_ids.append(_id)
                            # if len(greenlist_ids) > green_ids_size:
                            #     break
                    for cluster_id in green_cluster_ids_npy:
                        vocab_id_list = np.where(self.kmeans_label == cluster_id)[0]
                        for vocab_id in vocab_id_list:
                            if vocab_id < self.vocab_size:
                                greenlist_ids.append(vocab_id)
                            if len(greenlist_ids) >= green_ids_size:
                                break
                        if len(greenlist_ids) >= green_ids_size:
                            break
                    print("len(greenlist_ids): ", len(greenlist_ids))
                    # print("check 021")
                    # print(greenlist_ids)
                    # check_id_dup = 0
                    # greenlist_ids_numpy = np.array(greenlist_ids)
                    # for i in range(self.vocab_size):
                    #     if i in greenlist_ids_numpy:
                    #         check_id_dup += 1
                    # print("check_id_dup: ", check_id_dup)
                    # from collections import Counter
                    # count = Counter(greenlist_ids)
                    # print("dup: ", [item for item, cnt in count.items() if cnt > 1])
                    greenlist_ids = torch.tensor(greenlist_ids, device=hidden_embeddings.device)

                    # print(greenlist_ids)
                    # print("check 03")
                    self.kmeans_green_cluster_lookup.append(greenlist_ids)
            else:
                self.kmeans_green_cluster_lookup = []
                self.cluster_num = np.max(self.kmeans_label)
                green_cluster_size = int(self.cluster_num * self.gamma)
                # green_ids_size = int(self.vocab_size * self.gamma)
                # print("green_cluster_size:", green_cluster_size)
                # print("check 1")
                for a in range(self.discrete_value_number+1):
                    # print(a)
                    prf_key = hashint(torch.tensor(self.hash_key * a)).item()
                    self.rng.manual_seed(prf_key % (2**64 - 1))
                    cluster_permutation = torch.randperm(
                        self.cluster_num, device=hidden_embeddings.device, generator=self.rng
                    )
                    # print("check 01")
                    if self.select_green_tokens:
                        green_cluster_ids = cluster_permutation[:green_cluster_size]
                    else:
                        green_cluster_ids = cluster_permutation[
                            (self.cluster_num - green_cluster_size) :
                        ]
                    greenlist_ids = []
                    # print("check 02")
                    green_cluster_ids_npy = green_cluster_ids.cpu().detach().numpy()
                    for _id in range(self.kmeans_label.shape[0]):
                        if self.kmeans_label[_id] in green_cluster_ids_npy:
                            greenlist_ids.append(_id)
                            # if len(greenlist_ids) > green_ids_size:
                            #     break
                    print("len(greenlist_ids): ", len(greenlist_ids))
                    # print("check 021")
                    greenlist_ids = torch.tensor(greenlist_ids, device=hidden_embeddings.device)
                    # print("check 03")
                    self.kmeans_green_cluster_lookup.append(greenlist_ids)

        # input("check -0")
        list_of_greenlist_ids = [None for _ in input_ids]  # Greenlists could differ in length
        list_of_prf_seed = []
        # import pdb; pdb.set_trace()
        for b_idx, (input_seq, hidden_embeddings_seq) in enumerate(zip(input_ids, hidden_embeddings)):
            if self.self_salt:
                # input("check call _score_rejection_sampling")
                greenlist_ids = self._score_rejection_sampling(input_seq, scores[b_idx])
            else:
                if "sem" in self.prf_type:
                    greenlist_ids, prf_seed = self._sem_get_greenlist_ids(hidden_embeddings_seq, self.cl_mlp)
                    list_of_prf_seed.append(prf_seed)
                else:
                    greenlist_ids = self._get_greenlist_ids(input_seq)
            list_of_greenlist_ids[b_idx] = greenlist_ids

            # logic for computing and storing spike entropies for analysis
            if self.store_spike_ents:
                if self.spike_entropies is None:
                    self.spike_entropies = [[] for _ in range(input_ids.shape[0])]
                self.spike_entropies[b_idx].append(self._compute_spike_entropy(scores[b_idx]))

        green_tokens_mask = self._calc_greenlist_mask(
            scores=scores, greenlist_token_ids=list_of_greenlist_ids
        )
        scores = self._bias_greenlist_logits(
            scores=scores, greenlist_mask=green_tokens_mask, greenlist_bias=self.delta
        )

        return scores, list_of_greenlist_ids, list_of_prf_seed


class WatermarkDetector(WatermarkBase):
    """This is the detector for all watermarks imprinted with WatermarkLogitsProcessor.

    The detector needs to be given the exact same settings that were given during text generation  to replicate the watermark
    greenlist generation and so detect the watermark.
    This includes the correct device that was used during text generation, the correct tokenizer, the correct
    seeding_scheme name, and parameters (delta, gamma).

    Optional arguments are
    * normalizers ["unicode", "homoglyphs", "truecase"] -> These can mitigate modifications to generated text that could trip the watermark
    * ignore_repeated_ngrams -> This option changes the detection rules to count every unique ngram only once.
    * z_threshold -> Changing this threshold will change the sensitivity of the detector.
    """

    def __init__(
        self,
        *args,
        device: torch.device = None,
        tokenizer: Tokenizer = None,
        z_threshold: float = 4.0,
        normalizers: list[str] = ["unicode"],  # or also: ["unicode", "homoglyphs", "truecase"]
        ignore_repeated_ngrams: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # also configure the metrics returned/preprocessing options
        assert device, "Must pass device"
        assert tokenizer, "Need an instance of the generating tokenizer to perform detection"

        self.tokenizer = tokenizer
        self.device = device
        self.z_threshold = z_threshold
        self.rng = torch.Generator(device=self.device)

        self.normalizers = []
        for normalization_strategy in normalizers:
            self.normalizers.append(normalization_strategy_lookup(normalization_strategy))
        self.ignore_repeated_ngrams = ignore_repeated_ngrams

    def dummy_detect(
        self,
        return_prediction: bool = True,
        return_scores: bool = True,
        z_threshold: float = None,
        return_num_tokens_scored: bool = True,
        return_num_green_tokens: bool = True,
        return_green_fraction: bool = True,
        return_green_token_mask: bool = False,
        return_all_window_scores: bool = False,
        return_z_score: bool = True,
        return_z_at_T: bool = True,
        return_p_value: bool = True,
    ):
        # HF-style output dictionary
        score_dict = dict()
        if return_num_tokens_scored:
            score_dict.update(dict(num_tokens_scored=float("nan")))
        if return_num_green_tokens:
            score_dict.update(dict(num_green_tokens=float("nan")))
        if return_green_fraction:
            score_dict.update(dict(green_fraction=float("nan")))
        if return_z_score:
            score_dict.update(dict(z_score=float("nan")))
        if return_p_value:
            z_score = score_dict.get("z_score")
            if z_score is None:
                z_score = float("nan")
            score_dict.update(dict(p_value=float("nan")))
        if return_green_token_mask:
            score_dict.update(dict(green_token_mask=[]))
        if return_all_window_scores:
            score_dict.update(dict(window_list=[]))
        if return_z_at_T:
            score_dict.update(dict(z_score_at_T=torch.tensor([])))

        output_dict = {}
        if return_scores:
            output_dict.update(score_dict)
        # if passed return_prediction then perform the hypothesis test and return the outcome
        if return_prediction:
            z_threshold = z_threshold if z_threshold else self.z_threshold
            assert (
                z_threshold is not None
            ), "Need a threshold in order to decide outcome of detection test"
            output_dict["prediction"] = False

        return output_dict

    def _compute_z_score(self, observed_count, T):
        # count refers to number of green tokens, T is total number of tokens
        expected_count = self.gamma
        numer = observed_count - expected_count * T
        denom = sqrt(T * expected_count * (1 - expected_count))
        z = numer / denom
        return z

    def _compute_p_value(self, z):
        p_value = scipy.stats.norm.sf(z)
        return p_value

    # @lru_cache(maxsize=2**32)
    def _get_ngram_score_cached(self, prefix: tuple[int], target: int):
        """Expensive re-seeding and sampling is cached."""
        # Handle with care, should ideally reset on __getattribute__ access to self.prf_type, self.context_width, self.self_salt, self.hash_key
        greenlist_ids = self._get_greenlist_ids(torch.as_tensor(prefix, device=self.device))
        return True if target in greenlist_ids else False

    def _score_ngrams_in_passage(self, input_ids: torch.Tensor):
        """Core function to gather all ngrams in the input and compute their watermark."""
        if len(input_ids) - self.context_width < 1:
            raise ValueError(
                f"Must have at least {1} token to score after "
                f"the first min_prefix_len={self.context_width} tokens required by the seeding scheme."
            )

        # Compute scores for all ngrams contexts in the passage:
        # rj since here they use the previous tokens only. They don't need to calculate the green list one by one. They can just evaluate one time and count the freq to get the overall results.
        token_ngram_generator = ngrams(
            input_ids.cpu().tolist(), self.context_width + 1 - self.self_salt
        )
        frequencies_table = collections.Counter(token_ngram_generator)
        ngram_to_watermark_lookup = {}
        for idx, ngram_example in enumerate(frequencies_table.keys()):
            prefix = ngram_example if self.self_salt else ngram_example[:-1]
            target = ngram_example[-1]
            ngram_to_watermark_lookup[ngram_example] = self._get_ngram_score_cached(prefix, target)

        output_TF_lookup = []
        targets_loopup = []
        token_ngram_generator = ngrams(
            input_ids.cpu().tolist(), self.context_width + 1 - self.self_salt
        )
        for idx, ngram_example in enumerate(token_ngram_generator):
            prefix = ngram_example if self.self_salt else ngram_example[:-1]
            target = ngram_example[-1]
            output_TF_lookup.append(self._get_ngram_score_cached(prefix, target))
            targets_loopup.append(target)

        return ngram_to_watermark_lookup, frequencies_table, output_TF_lookup, targets_loopup

    def _get_green_at_T_booleans(self, input_ids, ngram_to_watermark_lookup) -> tuple[torch.Tensor]:
        """Generate binary list of green vs. red per token, a separate list that ignores repeated ngrams, and a list of offsets to
        convert between both representations:
        green_token_mask = green_token_mask_unique[offsets] except for all locations where otherwise a repeat would be counted # rj I cannot understand this.
        """
        green_token_mask, green_token_mask_unique, offsets = [], [], []
        used_ngrams = {}
        unique_ngram_idx = 0
        ngram_examples = ngrams(input_ids.cpu().tolist(), self.context_width + 1 - self.self_salt)

        for idx, ngram_example in enumerate(ngram_examples):
            green_token_mask.append(ngram_to_watermark_lookup[ngram_example])
            if self.ignore_repeated_ngrams:
                if ngram_example in used_ngrams:
                    pass
                else:
                    used_ngrams[ngram_example] = True
                    unique_ngram_idx += 1
                    green_token_mask_unique.append(ngram_to_watermark_lookup[ngram_example])
            else:
                green_token_mask_unique.append(ngram_to_watermark_lookup[ngram_example])
                unique_ngram_idx += 1
            offsets.append(unique_ngram_idx - 1)
        return (
            torch.tensor(green_token_mask),
            torch.tensor(green_token_mask_unique),
            torch.tensor(offsets),
        )

    def _score_sequence(
        self,
        input_ids: torch.Tensor,
        return_num_tokens_scored: bool = True,
        return_num_green_tokens: bool = True,
        return_green_fraction: bool = True,
        return_green_token_mask: bool = False,
        return_z_score: bool = True,
        return_z_at_T: bool = True,
        return_p_value: bool = True,
    ):
        ngram_to_watermark_lookup, frequencies_table, output_TF_lookup, targets_loopup = self._score_ngrams_in_passage(input_ids)
        green_token_mask, green_unique, offsets = self._get_green_at_T_booleans(
            input_ids, ngram_to_watermark_lookup
        )

        # input("check here")

        # Count up scores over all ngrams
        if self.ignore_repeated_ngrams:
            # Method that only counts a green/red hit once per unique ngram.
            # New num total tokens scored (T) becomes the number unique ngrams.
            # We iterate over all unqiue token ngrams in the input, computing the greenlist
            # induced by the context in each, and then checking whether the last
            # token falls in that greenlist.
            num_tokens_scored = len(frequencies_table.keys())
            green_token_count = sum(ngram_to_watermark_lookup.values())
        else:
            num_tokens_scored = sum(frequencies_table.values())
            assert num_tokens_scored == len(input_ids) - self.context_width + self.self_salt
            green_token_count = sum(
                freq * outcome
                for freq, outcome in zip(
                    frequencies_table.values(), ngram_to_watermark_lookup.values()
                )
            )
        assert green_token_count == green_unique.sum()

        # HF-style output dictionary
        score_dict = dict()
        score_dict.update(dict(ngram_to_watermark_lookup=output_TF_lookup))
        score_dict.update(dict(targets_loopup=targets_loopup))
        if return_num_tokens_scored:
            score_dict.update(dict(num_tokens_scored=num_tokens_scored))
        if return_num_green_tokens:
            score_dict.update(dict(num_green_tokens=green_token_count))
        if return_green_fraction:
            score_dict.update(dict(green_fraction=(green_token_count / num_tokens_scored)))
        if return_z_score:
            score_dict.update(
                dict(z_score=self._compute_z_score(green_token_count, num_tokens_scored))
            )
        if return_p_value:
            z_score = score_dict.get("z_score")
            if z_score is None:
                z_score = self._compute_z_score(green_token_count, num_tokens_scored)
            score_dict.update(dict(p_value=self._compute_p_value(z_score)))
        if return_green_token_mask:
            score_dict.update(dict(green_token_mask=green_token_mask.tolist()))
        if return_z_at_T:
            # Score z_at_T separately:
            sizes = torch.arange(1, len(green_unique) + 1)
            seq_z_score_enum = torch.cumsum(green_unique, dim=0) - self.gamma * sizes
            seq_z_score_denom = torch.sqrt(sizes * self.gamma * (1 - self.gamma))
            z_score_at_effective_T = seq_z_score_enum / seq_z_score_denom
            z_score_at_T = z_score_at_effective_T[offsets]
            assert torch.isclose(z_score_at_T[-1], torch.tensor(z_score))

            score_dict.update(dict(z_score_at_T=z_score_at_T))

        return score_dict

    def _score_windows_impl_batched(
        self,
        input_ids: torch.Tensor,
        window_size: str,
        window_stride: int = 1,
    ):
        # Implementation details:
        # 1) --ignore_repeated_ngrams is applied globally, and windowing is then applied over the reduced binary vector
        #      this is only one way of doing it, another would be to ignore bigrams within each window (maybe harder to parallelize that)
        # 2) These windows on the binary vector of green/red hits, independent of context_width, in contrast to Kezhi's first implementation
        # 3) z-scores from this implementation cannot be directly converted to p-values, and should only be used as labels for a
        #    ROC chart that calibrates to a chosen FPR. Due, to windowing, the multiple hypotheses will increase scores across the board#
        #    naive_count_correction=True is a partial remedy to this

        ngram_to_watermark_lookup, frequencies_table = self._score_ngrams_in_passage(input_ids)
        green_mask, green_ids, offsets = self._get_green_at_T_booleans(
            input_ids, ngram_to_watermark_lookup
        )
        len_full_context = len(green_ids)

        partial_sum_id_table = torch.cumsum(green_ids, dim=0)

        if window_size == "max":
            # could start later, small window sizes cannot generate enough power
            # more principled: solve (T * Spike_Entropy - g * T) / sqrt(T * g * (1 - g)) = z_thresh for T
            sizes = range(1, len_full_context)
        else:
            sizes = [int(x) for x in window_size.split(",") if len(x) > 0]

        z_score_max_per_window = torch.zeros(len(sizes))
        cumulative_eff_z_score = torch.zeros(len_full_context)
        s = window_stride

        window_fits = False
        for idx, size in enumerate(sizes):
            if size <= len_full_context:
                # Compute hits within window for all positions in parallel:
                window_score = torch.zeros(len_full_context - size + 1, dtype=torch.long)
                # Include 0-th window
                window_score[0] = partial_sum_id_table[size - 1]
                # All other windows from the 1st:
                window_score[1:] = partial_sum_id_table[size::s] - partial_sum_id_table[:-size:s]

                # Now compute batched z_scores
                batched_z_score_enum = window_score - self.gamma * size
                z_score_denom = sqrt(size * self.gamma * (1 - self.gamma))
                batched_z_score = batched_z_score_enum / z_score_denom

                # And find the maximal hit
                maximal_z_score = batched_z_score.max()
                z_score_max_per_window[idx] = maximal_z_score

                z_score_at_effective_T = torch.cummax(batched_z_score, dim=0)[0]
                cumulative_eff_z_score[size::s] = torch.maximum(
                    cumulative_eff_z_score[size::s], z_score_at_effective_T[:-1]
                )
                window_fits = True  # successful computation for any window in sizes

        if not window_fits:
            raise ValueError(
                f"Could not find a fitting window with window sizes {window_size} for (effective) context length {len_full_context}."
            )

        # Compute optimal window size and z-score
        cumulative_z_score = cumulative_eff_z_score[offsets]
        optimal_z, optimal_window_size_idx = z_score_max_per_window.max(dim=0)
        optimal_window_size = sizes[optimal_window_size_idx]
        return (
            optimal_z,
            optimal_window_size,
            z_score_max_per_window,
            cumulative_z_score,
            green_mask,
        )

    def _score_sequence_window(
        self,
        input_ids: torch.Tensor,
        return_num_tokens_scored: bool = True,
        return_num_green_tokens: bool = True,
        return_green_fraction: bool = True,
        return_green_token_mask: bool = False,
        return_z_score: bool = True,
        return_z_at_T: bool = True,
        return_p_value: bool = True,
        window_size: str = None,
        window_stride: int = 1,
    ):
        (
            optimal_z,
            optimal_window_size,
            _,
            z_score_at_T,
            green_mask,
        ) = self._score_windows_impl_batched(input_ids, window_size, window_stride)

        # HF-style output dictionary
        score_dict = dict()
        if return_num_tokens_scored:
            score_dict.update(dict(num_tokens_scored=optimal_window_size))

        denom = sqrt(optimal_window_size * self.gamma * (1 - self.gamma))
        green_token_count = int(optimal_z * denom + self.gamma * optimal_window_size)
        green_fraction = green_token_count / optimal_window_size
        if return_num_green_tokens:
            score_dict.update(dict(num_green_tokens=green_token_count))
        if return_green_fraction:
            score_dict.update(dict(green_fraction=green_fraction))
        if return_z_score:
            score_dict.update(dict(z_score=optimal_z))
        if return_z_at_T:
            score_dict.update(dict(z_score_at_T=z_score_at_T))
        if return_p_value:
            z_score = score_dict.get("z_score", optimal_z)
            score_dict.update(dict(p_value=self._compute_p_value(z_score)))

        # Return per-token results for mask. This is still the same, just scored by windows
        # todo would be to mark the actually counted tokens differently
        if return_green_token_mask:
            score_dict.update(dict(green_token_mask=green_mask.tolist()))

        return score_dict

    def detect(
        self,
        text: str = None,
        tokenized_text: list[int] = None,
        window_size: str = None,
        window_stride: int = None,
        return_prediction: bool = True,
        return_scores: bool = True,
        z_threshold: float = None,
        convert_to_float: bool = False,
        **kwargs,
    ) -> dict:
        """Scores a given string of text and returns a dictionary of results."""

        assert (text is not None) ^ (
            tokenized_text is not None
        ), "Must pass either the raw or tokenized string"
        if return_prediction:
            kwargs[
                "return_p_value"
            ] = True  # to return the "confidence":=1-p of positive detections

        # run optional normalizers on text
        for normalizer in self.normalizers:
            text = normalizer(text)
        if len(self.normalizers) > 0:
            print(f"Text after normalization:\n\n{text}\n")

        if tokenized_text is None:
            assert self.tokenizer is not None, (
                "Watermark detection on raw string ",
                "requires an instance of the tokenizer ",
                "that was used at generation time.",
            )
            tokenized_text = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)[
                "input_ids"
            ][0].to(self.device)
            if tokenized_text[0] == self.tokenizer.bos_token_id:
                tokenized_text = tokenized_text[1:]
        else:
            # try to remove the bos_tok at beginning if it's there
            if (self.tokenizer is not None) and (tokenized_text[0] == self.tokenizer.bos_token_id):
                tokenized_text = tokenized_text[1:]

        # call score method
        output_dict = {}

        # print(window_size)
        # import pdb; pdb.set_trace()
        if window_size is not None:
            # assert window_size <= len(tokenized_text) cannot assert for all new types
            # print(window_size)
            # import pdb; pdb.set_trace()
            score_dict = self._score_sequence_window(
                tokenized_text,
                window_size=window_size,
                window_stride=window_stride,
                **kwargs,
            )
            output_dict.update(score_dict)
        else:
            score_dict = self._score_sequence(tokenized_text, **kwargs)
        if return_scores:
            output_dict.update(score_dict)
        # if passed return_prediction then perform the hypothesis test and return the outcome
        if return_prediction:
            z_threshold = z_threshold if z_threshold else self.z_threshold
            assert (
                z_threshold is not None
            ), "Need a threshold in order to decide outcome of detection test"
            output_dict["prediction"] = score_dict["z_score"] > z_threshold
            if output_dict["prediction"]:
                output_dict["confidence"] = 1 - score_dict["p_value"]

        # convert any numerical values to float if requested
        if convert_to_float:
            for key, value in output_dict.items():
                if isinstance(value, int):
                    output_dict[key] = float(value)

        return output_dict

class SemWatermarkDetector(WatermarkDetector):

    def __init__(
        self,
        *args,
        decoder,
        cl_mlp,
        cl_discrete_value_number,
        cl_mean_pooling,
        cl_pooling_method,
        cl_k,
        flag_kmeans,
        kmeans_label_file = "kmean",
        device: torch.device = None,
        tokenizer: Tokenizer = None,
        z_threshold: float = 4.0,
        normalizers: list[str] = ["unicode"],  # or also: ["unicode", "homoglyphs", "truecase"]
        ignore_repeated_ngrams: bool = False,
        **kwargs,
    ):
        super().__init__(*args, device=device, tokenizer=tokenizer, z_threshold=z_threshold, normalizers=normalizers, ignore_repeated_ngrams=ignore_repeated_ngrams, **kwargs)
        # also configure the metrics returned/preprocessing options
        assert device, "Must pass device"
        assert tokenizer, "Need an instance of the generating tokenizer to perform detection"

        self.decoder = decoder
        self.cl_mlp = cl_mlp
        self.cl_discrete_value_number = cl_discrete_value_number
        self.cl_mean_pooling = cl_mean_pooling
        self.cl_pooling_method = cl_pooling_method
        self.cl_k = cl_k

        self.flag_kmeans = flag_kmeans
        self.kmeans_green_cluster_lookup = None
        if self.flag_kmeans:
            self.kmeans_label = np.load(f"./results/{kmeans_label_file}.npy")

        self.greenlist_table = {}
        
        if self.kmeans_green_cluster_lookup is None and self.flag_kmeans:
            # print("check")
            # input("check")
            yuliang = True #TODO
            if yuliang:
                self.kmeans_green_cluster_lookup = []
                self.cluster_num = np.max(self.kmeans_label)
                green_cluster_size = int(self.cluster_num * (self.gamma+0.1))
                green_ids_size = int(self.vocab_size * self.gamma)
                # print("check 1")
                for a in range(self.cl_discrete_value_number+1):
                    # print(a)
                    prf_key = hashint(torch.tensor(self.hash_key * a)).item()
                    self.rng.manual_seed(prf_key % (2**64 - 1))
                    cluster_permutation = torch.randperm(
                        self.cluster_num, device=device, generator=self.rng
                    )
                    # print("check 01")
                    if self.select_green_tokens:
                        green_cluster_ids = cluster_permutation[:green_cluster_size]
                    else:
                        green_cluster_ids = cluster_permutation[
                            (self.cluster_num - green_cluster_size) :
                        ]
                    greenlist_ids = []
                    # print("check 02")
                    green_cluster_ids_npy = green_cluster_ids.cpu().detach().numpy()
                    # for _id in range(self.kmeans_label.shape[0]):
                    #     if self.kmeans_label[_id] in green_cluster_ids_npy:
                    #         greenlist_ids.append(_id)
                    #         # if len(greenlist_ids) >= green_ids_size:
                    #         #     break
                    # print("check 021")
                    for cluster_id in green_cluster_ids_npy:
                        vocab_id_list = np.where(self.kmeans_label == cluster_id)[0]
                        for vocab_id in vocab_id_list:
                            if vocab_id < self.vocab_size:
                                greenlist_ids.append(vocab_id)
                            if len(greenlist_ids) >= green_ids_size:
                                break
                        if len(greenlist_ids) >= green_ids_size:
                            break
                    print("len(greenlist_ids): ", len(greenlist_ids))
                    greenlist_ids = torch.tensor(greenlist_ids, device=device)
                    # print("check 03")
                    print("len(greenlist_ids): ", len(greenlist_ids))
                    self.kmeans_green_cluster_lookup.append(greenlist_ids)
            else:
                self.kmeans_green_cluster_lookup = []
                self.cluster_num = np.max(self.kmeans_label)
                green_cluster_size = int(self.cluster_num * self.gamma)
                # green_ids_size = int(self.vocab_size * self.gamma)
                # print("check 1")
                for a in range(self.cl_discrete_value_number+1):
                    # print(a)
                    prf_key = hashint(torch.tensor(self.hash_key * a)).item()
                    self.rng.manual_seed(prf_key % (2**64 - 1))
                    cluster_permutation = torch.randperm(
                        self.cluster_num, device=device, generator=self.rng
                    )
                    # print("check 01")
                    if self.select_green_tokens:
                        green_cluster_ids = cluster_permutation[:green_cluster_size]
                    else:
                        green_cluster_ids = cluster_permutation[
                            (self.cluster_num - green_cluster_size) :
                        ]
                    greenlist_ids = []
                    # print("check 02")
                    green_cluster_ids_npy = green_cluster_ids.cpu().detach().numpy()
                    for _id in range(self.kmeans_label.shape[0]):
                        if self.kmeans_label[_id] in green_cluster_ids_npy:
                            greenlist_ids.append(_id)
                            # if len(greenlist_ids) >= green_ids_size:
                            #     break
                    # print("check 021")
                    greenlist_ids = torch.tensor(greenlist_ids, device=device)
                    # print("check 03")
                    print("len(greenlist_ids): ", len(greenlist_ids))
                    self.kmeans_green_cluster_lookup.append(greenlist_ids)

        # import pdb ; pdb.set_trace()

        # self.tokenizer = tokenizer
        # self.device = device
        # self.z_threshold = z_threshold
        # self.rng = torch.Generator(device=self.device)

        # self.normalizers = []
        # for normalization_strategy in normalizers:
        #     self.normalizers.append(normalization_strategy_lookup(normalization_strategy))
        # self.ignore_repeated_ngrams = ignore_repeated_ngrams

    def _sem_seed_rng(self, hidden_embeddings: torch.LongTensor, cl_mlp) -> None:
        """Seed RNG from local context. Not batched, because the generators we use (like cuda.random) are not batched."""
        # Need to have enough context for seed generation
        # if input_ids.shape[-1] < self.context_width:
        #     raise ValueError(
        #         f"seeding_scheme requires at least a {self.context_width} token prefix to seed the RNG."
        #     )


        if "sem" in self.prf_type:
            prf_key, a = prf_lookup[self.prf_type](
                    hidden_embeddings=F.normalize(hidden_embeddings, p=2, dim=1), salt_key=self.hash_key, cl_mlp=cl_mlp, discrete_value_number=self.cl_discrete_value_number
                )
        else:
            prf_key = prf_lookup[self.prf_type](
                    hidden_embeddings=F.normalize(hidden_embeddings, p=2, dim=1), salt_key=self.hash_key, cl_mlp=cl_mlp
                )
        # print("check2")
        # enable for long, interesting streams of pseudorandom numbers: print(prf_key)
        self.rng.manual_seed(prf_key % (2**64 - 1))  # safeguard against overflow from long

        return a

    def _sem_get_greenlist_ids(self, hidden_embeddings: torch.LongTensor, cl_mlp) -> torch.LongTensor:
        """Seed rng based on local context width and use this information to generate ids on the green list."""
        a = self._sem_seed_rng(hidden_embeddings=hidden_embeddings, cl_mlp=cl_mlp)

        if self.flag_kmeans:
            greenlist_ids = self.kmeans_green_cluster_lookup[a].detach()
        else:
            greenlist_size = int(self.vocab_size * self.gamma)
            vocab_permutation = torch.randperm(
                self.vocab_size, device=hidden_embeddings.device, generator=self.rng
            )
            if self.select_green_tokens:  # directly
                greenlist_ids = vocab_permutation[:greenlist_size]  # new
            else:  # select green via red
                greenlist_ids = vocab_permutation[
                    (self.vocab_size - greenlist_size) :
                ]  # legacy behavior
        return greenlist_ids, a

    # @lru_cache(maxsize=2**32)
    def _get_ngram_score_cached(self, prefix: torch.Tensor, target: int, past_key_values, cache_last_hidden_emb=None):
        """Expensive re-seeding and sampling is cached."""
        # Handle with care, should ideally reset on __getattribute__ access to self.prf_type, self.context_width, self.self_salt, self.hash_key
        # print(prefix.shape)
        # input("check prefix.shape")
        # attention_mask = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        #  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        #  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        #  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        #  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        #  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        #  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        #  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        #  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        #  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        #  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        #  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        #  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]).cuda()

        with torch.no_grad():
            if past_key_values is not None:
                output = self.decoder(prefix, past_key_values=past_key_values, use_cache=True, return_dict=True)
            else:
                output = self.decoder(prefix, use_cache=True, return_dict=True)

        torch.cuda.empty_cache()
        # cache_last_hidden_emb.append(outputs.hidden_states[-1])
        all_last_hidden_emb = torch.cat(cache_last_hidden_emb + [output.last_hidden_state], dim=1)
        # TODO: this is detection stage
        # if self.cl_mean_pooling == 
        # mean_last_hidden_emb = torch.mean(all_last_hidden_emb, dim=1, keepdim=True)

        if self.cl_mean_pooling:
            if self.cl_pooling_method == 'mean':
                mean_last_hidden_emb = torch.mean(all_last_hidden_emb, dim=1, keepdim=True)
            elif self.cl_pooling_method == 'former_k':
                # pdb.set_trace()
                k = self.cl_k
                start_id = max(0, all_last_hidden_emb.shape[1] - k)
                mean_last_hidden_emb = torch.mean(all_last_hidden_emb[:, start_id: , :], dim=1, keepdim=True)
            elif self.cl_pooling_method == 'weighted_former_k':
                k = self.cl_k
                start_id = max(0, all_last_hidden_emb.shape[1] - k)
                accumulated_emb = 0
                all_weight_count = 0
                for j in range(start_id, all_last_hidden_emb.shape[1]):
                    weight = j - start_id + 1 + k // 2
                    all_weight_count += weight
                    accumulated_emb += all_last_hidden_emb[:, j:j+1, :] * weight
                mean_last_hidden_emb = accumulated_emb / all_weight_count
            else:
                raise("Not emplemented method.")
        else:
            raise("No pooling is not emplemented.")

        greenlist_ids, a = self._sem_get_greenlist_ids(mean_last_hidden_emb[0, :, :], self.cl_mlp)
        # import pdb; pdb.set_trace()
        current_token_result = True if target in greenlist_ids else False
        return current_token_result, output.past_key_values, a, output.last_hidden_state

    def _score_ngrams_in_passage(self, input_ids: torch.Tensor, tokenized_prompt):
        """Core function to gather all ngrams in the input and compute their watermark."""
        if len(input_ids) - self.context_width < 1:
            raise ValueError(
                f"Must have at least {1} token to score after "
                f"the first min_prefix_len={self.context_width} tokens required by the seeding scheme."
            )
        ngram_to_watermark_lookup = []
        detect_seed = []
        past_key_values = None
        cache_last_hidden_emb = []
        targets_loopup = []
        for idx in range(len(input_ids)):
            torch.cuda.empty_cache()
            if idx > 0:
                prefix = input_ids[idx - 1:idx].unsqueeze(0)
            else:
                prefix = tokenized_prompt.unsqueeze(0)
            target = input_ids[idx]
            targets_loopup.append(target)
            outputs = self._get_ngram_score_cached(prefix, target, past_key_values, cache_last_hidden_emb=cache_last_hidden_emb)
            del past_key_values
            past_key_values = outputs[1]
            detect_seed.append(outputs[2])
            cache_last_hidden_emb.append(outputs[3])

            ngram_to_watermark_lookup.append(outputs[0])

        return ngram_to_watermark_lookup, detect_seed, targets_loopup

    def _score_sequence(
        self,
        input_ids: torch.Tensor,
        tokenized_prompt = None,
        return_num_tokens_scored: bool = True,
        return_num_green_tokens: bool = True,
        return_green_fraction: bool = True,
        return_green_token_mask: bool = False,
        return_z_score: bool = True,
        return_z_at_T: bool = True,
        return_p_value: bool = True,
    ):
        ngram_to_watermark_lookup, detect_seed, targets_loopup = self._score_ngrams_in_passage(input_ids, tokenized_prompt=tokenized_prompt)
        num_tokens_scored = len(ngram_to_watermark_lookup)
        assert num_tokens_scored == len(input_ids) - self.context_width + self.self_salt + 1
        green_token_count = sum(ngram_to_watermark_lookup)

        # HF-style output dictionary
        score_dict = dict()
        score_dict.update(dict(detect_seed=detect_seed))
        score_dict.update(dict(ngram_to_watermark_lookup=ngram_to_watermark_lookup))
        score_dict.update(dict(targets_loopup=targets_loopup))
        if return_num_tokens_scored:
            score_dict.update(dict(num_tokens_scored=num_tokens_scored))
        if return_num_green_tokens:
            score_dict.update(dict(num_green_tokens=green_token_count))
        if return_green_fraction:
            score_dict.update(dict(green_fraction=(green_token_count / num_tokens_scored)))
        if return_z_score:
            score_dict.update(
                dict(z_score=self._compute_z_score(green_token_count, num_tokens_scored))
            )
        if return_p_value:
            z_score = score_dict.get("z_score")
            if z_score is None:
                z_score = self._compute_z_score(green_token_count, num_tokens_scored)
            score_dict.update(dict(p_value=self._compute_p_value(z_score)))
        if return_green_token_mask: 
            raise("Please make sure return_green_token_mask is False")
            score_dict.update(dict(green_token_mask=green_token_mask.tolist()))
        if return_z_at_T:
            raise("Not emplemented.")
            # Score z_at_T separately:
            sizes = torch.arange(1, len(green_unique) + 1)
            seq_z_score_enum = torch.cumsum(green_unique, dim=0) - self.gamma * sizes
            seq_z_score_denom = torch.sqrt(sizes * self.gamma * (1 - self.gamma))
            z_score_at_effective_T = seq_z_score_enum / seq_z_score_denom
            z_score_at_T = z_score_at_effective_T[offsets]
            assert torch.isclose(z_score_at_T[-1], torch.tensor(z_score))

            score_dict.update(dict(z_score_at_T=z_score_at_T))

        return score_dict

    def detect(
        self,
        text: str = None,
        prompt = None,
        prompt_token = None,
        tokenized_text: list[int] = None,
        window_size: str = None,
        window_stride: int = None,
        return_prediction: bool = True,
        return_scores: bool = True,
        z_threshold: float = None,
        convert_to_float: bool = False,
        **kwargs,
    ) -> dict:
        """Scores a given string of text and returns a dictionary of results."""

        assert (text is not None) ^ (
            tokenized_text is not None
        ), "Must pass either the raw or tokenized string"
        if return_prediction:
            kwargs[
                "return_p_value"
            ] = True  # to return the "confidence":=1-p of positive detections

        # run optional normalizers on text
        for normalizer in self.normalizers:
            text = normalizer(text)
        if len(self.normalizers) > 0:
            print(f"Text after normalization:\n\n{text}\n")

        if tokenized_text is None:
            assert self.tokenizer is not None, (
                "Watermark detection on raw string ",
                "requires an instance of the tokenizer ",
                "that was used at generation time.",
            )
            tokenized_text = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)[
                "input_ids"
            ][0].to(self.device)
            tokenized_prompt = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)[
                "input_ids"
            ][0].to(self.device)

            if tokenized_text[0] == self.tokenizer.bos_token_id:
                tokenized_text = tokenized_text[1:]
            if tokenized_prompt[0] == self.tokenizer.bos_token_id:
                tokenized_prompt = tokenized_prompt[1:]
        else:
            # input("check tokenizer")
            # try to remove the bos_tok at beginning if it's there
            if (self.tokenizer is not None) and (tokenized_text[0] == self.tokenizer.bos_token_id):
                tokenized_text = tokenized_text[1:]

        bos_token_id_pos = 0
        for _i in range(len(prompt_token)):
            if prompt_token[_i] == self.tokenizer.pad_token_id:
                pass
            else:
                bos_token_id_pos = _i
                break
        prompt_token = prompt_token[bos_token_id_pos:]
        prompt_token = tokenized_prompt

        # call score method
        output_dict = {}

        if window_size is not None:
            # assert window_size <= len(tokenized_text) cannot assert for all new types
            score_dict = self._score_sequence_window(
                tokenized_text,
                window_size=window_size,
                window_stride=window_stride,
                **kwargs,
            )
            output_dict.update(score_dict)
        else:
            score_dict = self._score_sequence(tokenized_text, tokenized_prompt=torch.tensor(prompt_token).to(self.device), **kwargs)
        if return_scores:
            output_dict.update(score_dict)
            output_dict.update({
                "evluation_prompt": tokenized_prompt.tolist(),
            })
        # if passed return_prediction then perform the hypothesis test and return the outcome
        if return_prediction:
            z_threshold = z_threshold if z_threshold else self.z_threshold
            assert (
                z_threshold is not None
            ), "Need a threshold in order to decide outcome of detection test"
            output_dict["prediction"] = score_dict["z_score"] > z_threshold
            if output_dict["prediction"]:
                output_dict["confidence"] = 1 - score_dict["p_value"]

        # convert any numerical values to float if requested
        if convert_to_float:
            for key, value in output_dict.items():
                if isinstance(value, int):
                    output_dict[key] = float(value)

        return output_dict

##########################################################################
# Ngram iteration from nltk, extracted to remove the dependency
# Natural Language Toolkit: Utility functions
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Steven Bird <stevenbird1@gmail.com>
#         Eric Kafe <kafe.eric@gmail.com> (acyclic closures)
# URL: <https://www.nltk.org/>
# For license information, see https://github.com/nltk/nltk/blob/develop/LICENSE.txt
##########################################################################


def ngrams(sequence, n, pad_left=False, pad_right=False, pad_symbol=None):
    sequence = iter(sequence)
    if pad_left:
        sequence = chain((pad_symbol,) * (n - 1), sequence)
    if pad_right:
        sequence = chain(sequence, (pad_symbol,) * (n - 1))
    iterables = tee(sequence, n)

    for i, sub_iterable in enumerate(iterables):  # For each window,
        for _ in range(i):  # iterate through every order of ngrams
            next(sub_iterable, None)  # generate the ngrams within the window.
    return zip(*iterables)  # Unpack and flattens the iterables.
