# -*- coding: utf-8 -*-
"""Selecao ANOVA com uma observacao por participante apenas no ranking.

As entradas e saidas continuam sendo gravacoes. Durante ``fit``, a media das
frases de cada pessoa e usada para calcular o F de ANOVA; durante ``transform``,
as colunas escolhidas sao aplicadas diretamente as gravacoes, sem agregacao.
Imputacao e remocao de constantes devem anteceder este seletor dentro do fold.

Exemplo em um sklearn Pipeline, sem metadata routing global::

    pipeline.fit(X_train, y_train, selector__groups=participants_train)

Os valores F sao um criterio de ordenacao exploratorio. Nao sao uma inferencia
de significancia corrigida pela busca, pelos folds ou pela selecao de atributos.
Nenhuma estatistica de ``transform`` e utilizada para reajustar a selecao.
"""
from __future__ import annotations

from numbers import Integral
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import f_classif
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class ParticipantMeanANOVA(TransformerMixin, BaseEstimator):
    """Escolhe ``k`` atributos pelo F calculado entre medias de participantes.

    ``groups`` deve identificar a pessoa de cada linha de treino, com um unico
    rotulo binario por pessoa. Cada pessoa contribui exatamente uma media para
    o ranking, mesmo quando o numero de frases difere. Isso nao altera os pesos
    do classificador e nao corrige por si so todas as diferencas entre tarefas.

    Empates exatos de F sao resolvidos pelo menor indice original da coluna.
    Constantes/estatisticas indefinidas recebem -inf; +inf de separacao perfeita
    e mantido. As colunas selecionadas preservam a ordem original em transform.
    """

    def __init__(self, k=128):
        self.k = k

    def fit(self, X, y, groups=None):
        """Aprende o ranking somente a partir das pessoas fornecidas no treino."""
        if isinstance(self.k, (bool, np.bool_)) or not isinstance(self.k, Integral) or self.k < 1:
            raise ValueError("k deve ser um inteiro positivo")
        if y is None:
            raise ValueError("y binario e obrigatorio")
        if np.asarray(y).ndim != 1:
            raise ValueError("y deve ter uma dimensao")
        array, labels = check_X_y(X, y, dtype=np.float64, ensure_min_samples=2)
        if not np.isin(labels, (0, 1)).all() or len(np.unique(labels)) != 2:
            raise ValueError("y deve conter exatamente as classes 0 e 1, sem ausentes")
        if groups is None:
            raise ValueError("groups com participant_id e obrigatorio no fit")
        group_array = np.asarray(groups)
        if group_array.ndim != 1 or len(group_array) != len(array):
            raise ValueError("groups deve ter uma dimensao e uma identidade por linha de treino")
        if pd.isna(group_array).any():
            raise ValueError("groups nao pode conter identidades ausentes")
        if any(isinstance(g, str) and not g.strip() for g in group_array):
            raise ValueError("groups nao pode conter identidades vazias")
        try:
            codes, identities = pd.factorize(group_array, sort=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("groups deve conter identidades escalares comparaveis") from exc
        n_people = len(identities)
        mean_features = np.zeros((n_people, array.shape[1]), dtype=np.float64)
        np.add.at(mean_features, codes, array)
        counts = np.bincount(codes, minlength=n_people)
        mean_features /= counts[:, None]
        person_labels = np.empty(n_people, dtype=np.int8)
        for i in range(n_people):
            unique_labels = np.unique(labels[codes == i])
            if len(unique_labels) != 1:
                raise ValueError("Participante com rotulos divergentes no treino")
            person_labels[i] = int(unique_labels[0])
        class_counts = np.bincount(person_labels, minlength=2)
        if class_counts.min() < 2:
            raise ValueError("ANOVA por participante exige pelo menos duas pessoas por classe")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            warnings.simplefilter("ignore", category=RuntimeWarning)
            statistic, _ = f_classif(mean_features, person_labels)
        statistic = np.asarray(statistic, dtype=np.float64)
        statistic[np.isnan(statistic)] = -np.inf
        # f_classif pode produzir um F negativo de arredondamento em constantes.
        constant_means = np.ptp(mean_features, axis=0) == 0
        statistic[constant_means] = -np.inf
        indices = np.arange(array.shape[1])
        ranking = np.lexsort((indices, -statistic))
        self.n_features_in_ = array.shape[1]
        self.n_participants_ = n_people
        self.participant_class_counts_ = class_counts.copy()
        self.effective_k_ = min(int(self.k), self.n_features_in_)
        self.scores_ = statistic
        self.ranking_indices_ = ranking
        self.selected_indices_ = np.sort(ranking[:self.effective_k_])
        if hasattr(self, "feature_names_in_"):
            del self.feature_names_in_
        if hasattr(X, "columns") and all(isinstance(c, str) for c in X.columns):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        """Aplica o ranking aprendido, sem exigir IDs nem rotulos de teste."""
        check_is_fitted(self, ("scores_", "selected_indices_", "n_features_in_"))
        # O ranking usa float64, mas a aplicacao preserva float32/float64 para
        # nao alterar a precisao do restante do pipeline ao trocar o seletor.
        array = check_array(X, dtype=(np.float64, np.float32))
        if array.shape[1] != self.n_features_in_:
            raise ValueError(f"Esperava {self.n_features_in_} atributos; recebi {array.shape[1]}")
        if hasattr(self, "feature_names_in_") and hasattr(X, "columns"):
            if not np.array_equal(np.asarray(X.columns, dtype=object), self.feature_names_in_):
                raise ValueError("Ordem ou nomes dos atributos diferem do treino")
        return array[:, self.selected_indices_]

    def get_support(self, indices=False):
        """Retorna mascara ou indices na mesma convencao do SelectKBest."""
        check_is_fitted(self, ("selected_indices_", "n_features_in_"))
        if indices:
            return self.selected_indices_.copy()
        support = np.zeros(self.n_features_in_, dtype=bool)
        support[self.selected_indices_] = True
        return support

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, ("selected_indices_", "n_features_in_"))
        if input_features is None:
            features = getattr(self, "feature_names_in_", np.asarray([
                f"x{i}" for i in range(self.n_features_in_)
            ], dtype=object))
        else:
            features = np.asarray(input_features, dtype=object)
            if features.ndim != 1 or len(features) != self.n_features_in_:
                raise ValueError("input_features deve corresponder aos atributos de treino")
            if hasattr(self, "feature_names_in_") and not np.array_equal(features, self.feature_names_in_):
                raise ValueError("input_features difere dos nomes de treino")
        return np.asarray(features, dtype=object)[self.selected_indices_]
