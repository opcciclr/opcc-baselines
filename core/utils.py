import logging
import os
import traceback
from typing import List

import numpy as np
import opcc
import pandas as pd
import torch
from rliable import metrics
from tqdm import tqdm


def log_traceback(ex, ex_traceback=None):
    if ex_traceback is None:
        ex_traceback = ex.__traceback__
    tb_lines = [line.rstrip('\n') for line in
                traceback.format_exception(ex.__class__, ex, ex_traceback)]
    return tb_lines


def init_logger(base_path: str, name: str, file_mode='w'):
    formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s]'
                                  '[%(filename)s>%(funcName)s] => %(message)s')
    file_path = os.path.join(base_path, name + '.log')
    logger = logging.getLogger(name)
    logging.getLogger().handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    handler = logging.FileHandler(file_path, mode=file_mode)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)


def evaluate_queries(env_name, queries, network, runs, batch_size,
                     device: str = 'cpu',
                     mixture: bool = False,
                     verbose: bool = False) -> pd.DataFrame:
    predict_df = pd.DataFrame()
    for (policy_a_id, policy_b_id), query_batch in tqdm(queries.items(),
                                                        desc='Policy Pairs'):

        # validate policies existence
        policy_a, _ = opcc.get_policy(*policy_a_id)
        policy_a = policy_a.to(device)
        policy_b, _ = opcc.get_policy(*policy_b_id)
        policy_b = policy_b.to(device)

        # query
        obs_a = query_batch['obs_a']
        obs_b = query_batch['obs_b']
        # open_loop_horizon = 1
        policy_horizon = query_batch['policy_horizon']

        horizons = np.array(policy_horizon) + 1

        # evaluate
        pred_a = np.zeros((len(obs_a), network.num_ensemble))
        pred_b = np.zeros((len(obs_b), network.num_ensemble))

        for horizon in tqdm(np.unique(horizons), desc='Horizons',
                            disable=verbose):
            _filter = horizons == horizon
            state_a = np.array(obs_a)[_filter]
            state_b = np.array(obs_b)[_filter]
            action_a = np.array(query_batch['action_a'])[_filter]
            action_b = np.array(query_batch['action_b'])[_filter]
            filter_policy_horizon = np.array(policy_horizon)[_filter]
            assert all(filter_policy_horizon[0]
                       == filter_policy_horizon)

            pred_a[_filter, :] = mc_return(network=network,
                                           init_obs=state_a,
                                           init_actions=action_a,
                                           policy=policy_a,
                                           policy_horizon=filter_policy_horizon[0],
                                           device=device,
                                           runs=runs,
                                           mixture=mixture,
                                           eval_batch_size=batch_size,
                                           mixture_seed=0)

            pred_b[_filter, :] = mc_return(network=network,
                                           init_obs=state_b,
                                           init_actions=action_b,
                                           policy=policy_b,
                                           policy_horizon=filter_policy_horizon[0],
                                           device=device,
                                           runs=runs,
                                           mixture=mixture,
                                           eval_batch_size=batch_size,
                                           mixture_seed=0)

            filter_target = np.array(query_batch['target'])[_filter]
            filter_return_a = np.array(query_batch['info']['return_a'])[_filter]
            filter_return_b = np.array(query_batch['info']['return_b'])[_filter]
            filter_query_idx = np.where(_filter)[0]

            filter_distance_a = {f_k:np.array(f_v)[_filter]
                                 for f_k,f_v in
                                 query_batch['info']
                                 ['dataset_distance_a'].items()}

            filter_distance_b = {f_k:np.array(f_v)[_filter]
                                 for f_k,f_v in
                                 query_batch['info']
                                 ['dataset_distance_b'].items()}
            # store in dataframe
            for idx in range(len(state_a)):
                _stat = {
                    # ground-truth info
                    **{'env_name': env_name,
                       'policy_a_id': policy_a_id[1],
                       'policy_b_id': policy_b_id[1],
                       'query_idx': filter_query_idx[idx],
                       'policy_ids': (policy_a_id[1], policy_b_id[1]),
                       **{'obs_a_{}'.format(s_idx): s_i
                          for s_idx,s_i in enumerate(state_a[idx])},
                       **{'obs_b_{}'.format(s_idx): s_i
                          for s_idx, s_i in enumerate(state_b[idx])},
                       'horizon': horizon,
                       'open_loop_horizon': 1,
                       'policy_horizon': filter_policy_horizon[0],
                       'target': filter_target[idx],
                       'return_a': filter_return_a[idx],
                       'return_b': filter_return_b[idx]},

                    # query-a predictions
                    **{'pred_a_{}'.format(e_i): pred_a[_filter, :][idx][e_i]
                       for e_i in range(network.num_ensemble)},
                    **{'pred_a_mean': pred_a[_filter, :][idx].mean(),
                       'pred_a_iqm': metrics.aggregate_iqm([pred_a[_filter, :][idx]]),
                       'pred_a_median': np.median(pred_a[_filter, :][idx]),
                       'pred_a_max': pred_a[_filter, :][idx].max(),
                       'pred_a_min': pred_a[_filter, :][idx].min()},

                    # query-b predictions
                    **{'pred_b_{}'.format(e_i): pred_b[_filter, :][idx][e_i]
                       for e_i in range(network.num_ensemble)},
                    **{'pred_b_mean': pred_b[_filter, :][idx].mean(),
                       'pred_b_median': np.median(pred_b[_filter, :][idx]),
                       'pred_b_iqm': metrics.aggregate_iqm([pred_b[_filter, :][idx]]),
                       'pred_b_max': pred_b[_filter, :][idx].max(),
                       'pred_b_min': pred_b[_filter, :][idx].min()},

                    # query distances
                    **{'distance_a-{}'.format(k): v[idx] for k, v in
                       filter_distance_a.items()},
                    **{'distance_b-{}'.format(k): v[idx] for k, v in
                       filter_distance_b.items()}
                }
                predict_df = predict_df.append(_stat, ignore_index=True)

    return predict_df


@torch.no_grad()
def mc_return(network, init_obs, init_actions, policy, policy_horizon: int,
              device: str = 'cpu', runs: int = 1, mixture: bool = False,
              eval_batch_size: int = 128, mixture_seed: int = 0,
              verbose: bool = False) -> List[float]:
    assert len(init_obs) == len(init_actions), 'batch size not same'
    batch_size, obs_size = init_obs.shape
    open_loop_horizon = 1
    _, action_size = init_actions.shape

    # repeat for ensemble size and runs
    init_obs = torch.DoubleTensor(init_obs).unsqueeze(1).unsqueeze(1)
    init_obs = init_obs.repeat(1, runs, network.num_ensemble, 1)
    init_obs = init_obs.flatten(0, 1)
    init_action = torch.DoubleTensor(init_actions).unsqueeze(1).unsqueeze(1)
    init_action = init_action.repeat(1, runs, network.num_ensemble, 1, 1)
    init_action = init_action.flatten(0, 1)

    returns = np.zeros((batch_size * runs, network.num_ensemble))

    for batch_idx in tqdm(range(0, returns.shape[0], eval_batch_size),
                          desc='Batch', disable=verbose):
        batch_end_idx = batch_idx + eval_batch_size

        # reset
        step_obs = init_obs[batch_idx:batch_end_idx].to(device)

        if mixture:
            network.enable_mixture(mixture_seed)

        # step
        horizon = policy_horizon + open_loop_horizon
        for step in range(horizon):
            if step < open_loop_horizon:
                step_action = init_action[batch_idx:batch_end_idx, :, step, :] \
                    .to(device)
            else:
                step_action = policy(step_obs)

            assert len(step_action.shape) == 3, \
                'expected (batch, ensemble,action)'
            step_obs, reward, done = network.step(step_obs, step_action)
            assert len(step_obs.shape) == 3, 'expected (batch, ensemble, obs)'

            # move to cpu for saving cuda memory
            reward = reward.cpu().detach().numpy()
            returns[batch_idx:batch_end_idx][~done] += reward[~done]

        if device == 'cuda':
            torch.cuda.empty_cache()

        if mixture:
            network.disable_mixture()

    returns = returns.reshape((batch_size, runs, network.num_ensemble))
    returns = returns.mean(1)
    return returns
