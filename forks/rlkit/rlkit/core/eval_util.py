"""
Common evaluation utilities.
"""

from collections import OrderedDict
from numbers import Number

import numpy as np
import torch

import forks.rlkit.rlkit.pythonplusplus as ppp


def get_generic_path_information(paths, stat_prefix=""):
    """
    Get an OrderedDict with a bunch of statistic names and values.
    """
    statistics = OrderedDict()
    returns = [sum(path["rewards"]) for path in paths]

    rewards = np.vstack([path["rewards"] for path in paths])
    statistics.update(
        create_stats_ordered_dict("Rewards", rewards, stat_prefix=stat_prefix)
    )
    statistics.update(
        create_stats_ordered_dict("Returns", returns, stat_prefix=stat_prefix)
    )
    actions = [path["actions"] for path in paths]
    if len(actions[0].shape) == 1:
        actions = np.hstack([path["actions"] for path in paths])
    else:
        actions = np.vstack([path["actions"] for path in paths])
    statistics.update(
        create_stats_ordered_dict("Actions", actions, stat_prefix=stat_prefix)
    )
    statistics["Num Paths"] = len(paths)
    statistics["Proportion exploration"] = sum(
        [sum(path["explored"]) for path in paths]
    )[0] / sum([len(path["explored"]) for path in paths])
    statistics[stat_prefix + "Average Returns"] = get_average_returns(paths)

    for info_key in ["agent_infos"]:
        if info_key in paths[0]:
            all_env_infos = [
                ppp.list_of_dicts__to__dict_of_lists(p[info_key]) for p in paths
            ]
            for k in all_env_infos[0].keys():
                final_ks = np.array([info[k][-1] for info in all_env_infos])
                first_ks = np.array([info[k][0] for info in all_env_infos])
                all_ks = np.concatenate([info[k] for info in all_env_infos])
                statistics.update(
                    create_stats_ordered_dict(
                        stat_prefix + k,
                        final_ks,
                        stat_prefix="{}/final/".format(info_key),
                    )
                )
                statistics.update(
                    create_stats_ordered_dict(
                        stat_prefix + k,
                        first_ks,
                        stat_prefix="{}/initial/".format(info_key),
                    )
                )
                statistics.update(
                    create_stats_ordered_dict(
                        stat_prefix + k, all_ks, stat_prefix="{}/".format(info_key)
                    )
                )

    return statistics


def get_action_histograms(paths, stat_prefix=""):
    """
    Get an OrderedDict with a bunch of statistic names and values.
    """
    # print(paths[0]['agent_infos'][0]['dist'].shape)
    if len(paths[0]["agent_infos"]) == 0:  # catching interim experience case
        return interim_action_histograms(paths)
    if "dist" not in paths[0]["agent_infos"][0]:
        return {}  # early exit for algos with no probabilities
    statistics = OrderedDict()
    # actions = np.concatenate([p["actions"] for p in paths])
    # if len(actions.shape) == 2:
    #     actions = actions[:, 0]
    # probs = np.concatenate([ai["probs"] for p in paths for ai in p["agent_infos"]])
    # for a in set(actions):
    #     statistics[str(a)] = probs[actions == a]
    if len(paths[0]["agent_infos"][0]["dist"].shape) == 1:
        dists = np.concatenate(
            [np.expand_dims(ai["dist"], 0) for p in paths for ai in p["agent_infos"]]
        )
    else:
        dists = np.concatenate([ai["dist"] for p in paths for ai in p["agent_infos"]])
    if len(dists.shape) == 1:
        statistics[0] = dists
    else:
        for i in range(dists.shape[1]):
            statistics[str(i)] = dists[:, i]
    return statistics


def interim_action_histograms(paths):
    statistics = OrderedDict()
    statistics[0] = [p["actions"][0] for p in paths]
    return statistics


def get_average_returns(paths):
    returns = [sum(path["rewards"]) for path in paths]
    return np.mean(returns)


def create_stats_ordered_dict(
    name, data, stat_prefix=None, always_show_all_stats=True, exclude_max_min=False
):
    if stat_prefix is not None:
        name = "{}{}".format(stat_prefix, name)
    if isinstance(data, Number):
        return OrderedDict({name: data})

    if len(data) == 0:
        return OrderedDict()

        ordered_dict = OrderedDict()
        for number, d in enumerate(data):
            sub_dict = create_stats_ordered_dict("{0}_{1}".format(name, number), d)
            ordered_dict.update(sub_dict)
        return ordered_dict

    if isinstance(data, list):
        try:
            iter(data[0])
        except TypeError:
            pass
        else:
            data = np.concatenate(data)

    if isinstance(data, np.ndarray) and isinstance(data[0], torch.Tensor):
        for i, d in enumerate(data):
            data[i] = d.item()

    if isinstance(data, np.ndarray) and data.size == 1 and not always_show_all_stats:
        return OrderedDict({name: float(data)})

    stats = OrderedDict(
        [(name + " Mean", np.mean(data)), (name + " Std", np.std(data))]
    )
    if not exclude_max_min:
        stats[name + " Max"] = np.max(data)
        stats[name + " Min"] = np.min(data)
    return stats
