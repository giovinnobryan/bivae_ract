# Copyright 2018 The Cornac Authors. All Rights Reserved.
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
# ============================================================================

import numpy as np

from cornac.models.recommender import Recommender
from cornac.utils.common import scale
from cornac.exception import ScoreException


class BiVAE_RaCT(Recommender):
    def __init__(
        self,
        name="BiVAE_RaCT",
        k=10,
        encoder_structure=[100,40],
        act_fn="sigmoid",
        likelihood="pois",
        n_epochs=200,
        batch_size=128,
        learning_rate=0.001,
        beta_kl=1.0,
        cap_priors={"user": False, "item": False},
        trainable=True,
        verbose=False,
        seed=None,
        use_gpu=True,
    ):
        Recommender.__init__(self, name=name, trainable=trainable, verbose=verbose)
        self.k = k
        self.encoder_structure = encoder_structure
        self.act_fn = act_fn
        self.likelihood = likelihood
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.beta_kl = beta_kl
        self.cap_priors = cap_priors
        self.seed = seed
        self.use_gpu = use_gpu

    def pre_train_actor(self, train_set, val_set=None):
       
        Recommender.fit(self, train_set, val_set)

        import torch
        from .bivae import BiVAE, learn

        self.device = (
            torch.device("cuda:0")
            if (self.use_gpu and torch.cuda.is_available())
            else torch.device("cpu")
        )

        if self.trainable:
            feature_dim = {"user": None, "item": None}
            if self.cap_priors.get("user", False):
                if train_set.user_feature is None:
                    raise ValueError(
                        "CAP priors for users is set to True but no user features are provided"
                    )
                else:
                    feature_dim["user"] = train_set.user_feature.feature_dim

            if self.cap_priors.get("item", False):
                if train_set.item_feature is None:
                    raise ValueError(
                        "CAP priors for items is set to True but no item features are provided"
                    )
                else:
                    feature_dim["item"] = train_set.item_feature.feature_dim

            if self.seed is not None:
                torch.manual_seed(self.seed)
                torch.cuda.manual_seed(self.seed)

            if not hasattr(self, "bivaecf"):
                num_items = train_set.matrix.shape[1]
                num_users = train_set.matrix.shape[0]
                self.bivae = BiVAE(
                    k=self.k,
                    user_encoder_structure=[num_items] + self.encoder_structure,
                    item_encoder_structure=[num_users] + self.encoder_structure,
                    act_fn=self.act_fn,
                    likelihood=self.likelihood,
                    cap_priors=self.cap_priors,
                    feature_dim=feature_dim,
                    batch_size=self.batch_size,
                ).to(self.device)

            learn(
                self.bivae,
                self.train_set,
                n_epochs=self.n_epochs,
                batch_size=self.batch_size,
                learn_rate=self.learning_rate,
                beta_kl=self.beta_kl,
                verbose=self.verbose,
                device=self.device,
            )

        elif self.verbose:
            print("%s is trained already (trainable = False)" % (self.name))

        return self
    
    def pre_train_critic:
        
        return self
    
    def actor_critic:
        
        return self
    

    def score(self, user_idx, item_idx=None):
        """Predict the scores/ratings of a user for an item.

        Parameters
        ----------
        user_idx: int, required
            The index of the user for whom to perform score prediction.

        item_idx: int, optional, default: None
            The index of the item for which to perform score prediction.
            If None, scores for all known items will be returned.

        Returns
        -------
        res : A scalar or a Numpy array
            Relative scores that the user gives to the item or to all known items

        """

        if item_idx is None:
            if self.train_set.is_unk_user(user_idx):
                raise ScoreException(
                    "Can't make score prediction for (user_id=%d)" % user_idx
                )

            theta_u = self.bivae.mu_theta[user_idx].view(1, -1)
            beta = self.bivae.mu_beta
            known_item_scores = (
                self.bivae.decode_user(theta_u, beta).cpu().numpy().ravel()
            )

            return known_item_scores
        else:
            if self.train_set.is_unk_user(user_idx) or self.train_set.is_unk_item(
                item_idx
            ):
                raise ScoreException(
                    "Can't make score prediction for (user_id=%d, item_id=%d)"
                    % (user_idx, item_idx)
                )

            theta_u = self.bivae.mu_theta[user_idx].view(1, -1)
            beta_i = self.bivae.mu_beta[item_idx].view(1, -1)
            pred = self.bivae.decode_user(theta_u, beta_i).cpu().numpy().ravel()

            pred = scale(
                pred, self.train_set.min_rating, self.train_set.max_rating, 0.0, 1.0
            )

            return pred
