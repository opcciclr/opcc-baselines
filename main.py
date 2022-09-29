import argparse
import json
import logging
import os
import pickle
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import opcc
import pandas as pd
import plotly.graph_objects as go
import torch
import wandb
from sklearn import metrics
from wandb.plot import scatter

from core.config import BaseConfig
from core.train import train_dynamics
from core.uncertainty import confidence_interval as ci
from core.uncertainty import ensemble_voting as ev
from core.utils import evaluate_queries
from core.utils import init_logger


def _seed(seed=0, cuda=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)


def get_args(arg_str: str = None):
    # gather arguments
    parser = argparse.ArgumentParser(prog='main.py',
                                     description='opcc-baselines',
                                     formatter_class=argparse.
                                     ArgumentDefaultsHelpFormatter)

    # job arguments
    job_args = parser.add_argument_group('job args')
    job_args.add_argument('--no-cuda', action='store_true',
                          help='no cuda usage')
    job_args.add_argument('--job', required=True,
                          choices=['train-dynamics',
                                   'evaluate-queries',
                                   'uncertainty-test'])
    # paths
    path_args = parser.add_argument_group('paths setup')
    path_args.add_argument('--result-dir', type=Path,
                           default=Path(os.path.join(os.getcwd(), 'results')),
                           help="directory to store results")
    # wandb setup
    wandb_args = parser.add_argument_group('wandb setup')
    wandb_args.add_argument('--wandb-project-name',
                            default='opcc-diverse-baselines',
                            help='name of the wandb project')
    wandb_args.add_argument('--use-wandb', action='store_true',
                            help='use Weight and bias visualization lib')
    wandb_args.add_argument('--wandb-dir', default=os.path.join('~/'),
                            help="directory Path to store wandb data")

    # dynamics resumption args
    dynamics_resume_args = parser.add_argument_group('args for resuming '
                                                     'dynamics training')
    dynamics_resume_args.add_argument('--resume', choices=['local', 'wandb'],
                                      help='resumes training')
    dynamics_resume_args.add_argument('--wandb-run-path',
                                      help="wandb run path for restoring "
                                           "dynamics training. It's used with "
                                           "{--resume wandb}")
    # dynamics args
    dynamics_args = parser.add_argument_group('args for training dynamics')
    dynamics_args.add_argument('--env-name', default='HalfCheetah-v2',
                               help='name of the environment')
    dynamics_args.add_argument('--dataset-name', default='random',
                               help='name of the dataset')
    dynamics_args.add_argument('--dynamics-type', default='feed-forward',
                               choices=['feed-forward', 'autoregressive'],
                               help='type of dynamics model')
    dynamics_args.add_argument('--deterministic', action='store_true',
                               help='if True, we use deterministic model '
                                    'otherwise stochastic')
    dynamics_args.add_argument('--dynamics-seed', default=0, type=int,
                               help='seed for training dynamics ')
    dynamics_args.add_argument('--log-interval', default=1, type=int,
                               help='log interval for training dynamics')
    dynamics_args.add_argument('--dynamics-checkpoint-interval', type=int,
                               default=1, help='update interval to save'
                                               ' dynamics checkpoint ')
    dynamics_args.add_argument('--hidden-size', type=int, default=200,
                               help='hidden size for Linear Layers ')

    dynamics_args.add_argument('--update-count', type=int, default=100,
                               help='epochs for training ')
    dynamics_args.add_argument('--dynamics-batch-size', type=int, default=256,
                               help='batch size for Dynamics Learning ')
    dynamics_args.add_argument('--reward-loss-coeff', type=int, default=1,
                               help='reward loss coefficient for training ')
    dynamics_args.add_argument('--observation-loss-coeff', type=int, default=1,
                               help='obs. loss coefficient for training ')
    dynamics_args.add_argument('--grad-clip-norm', type=float, default=5.0,
                               help='gradient clipping norm')
    dynamics_args.add_argument('--dynamics-lr', type=float, default=1e-3,
                               help='learning rate for Dynamics')
    dynamics_args.add_argument('--normalize', action='store_true',
                               help='normalizes the network inputs'
                                    ' with 0 mean and 1 std')
    dynamics_args.add_argument('--num-ensemble', default=1, type=int,
                               help='number of dynamics for ensemble ')
    dynamics_args.add_argument('--constant-prior-scale', type=float,
                               default=0, help='scale for constant priors')

    # queries evaluation args
    queries_args = parser.add_argument_group('args for evaluating queries')
    queries_args.add_argument('--restore-dynamics-from-wandb',
                              action='store_true',
                              help='restore model from wandb run')
    queries_args.add_argument('--wandb-dynamics-run-path', type=str,
                              help='wandb run id if restoring model')
    queries_args.add_argument('--mixture', action='store_true',
                              help='if enabled, randomly select a ensemble'
                                   'models at each step of query evaluation')
    queries_args.add_argument('--eval-runs', type=int, default=1,
                              help='run count for each query evaluation')
    queries_args.add_argument('--eval-batch-size', type=int, default=128,
                              help='batch size for query evaluation')
    queries_args.add_argument('--clip-obs', action='store_true',
                              help='clip the observation space with bounds for'
                                   ' query evaluation')
    queries_args.add_argument('--clip-reward', action='store_true',
                              help='clip the reward with dataset bounds'
                                   ' for query evaluation')

    # uncertainty-test arguments
    uncertain_args = parser.add_argument_group('args for uncertainty-test')
    uncertain_args.add_argument('--uncertainty-test-type',
                                default='ensemble-voting',
                                choices=['paired-confidence-interval',
                                         'unpaired-confidence-interval',
                                         'ensemble-voting'],
                                help='type of uncertainty test')
    uncertain_args.add_argument('--restore-query-eval-data-from-wandb',
                                action='store_true',
                                help='get query evaluation data from wandb')
    uncertain_args.add_argument('--wandb-query-eval-data-run-path', type=str,
                                help='wandb run id  having query eval data')

    # Process arguments
    args = parser.parse_args(arg_str.split(" ") if arg_str else None)
    args.device = ('cuda' if (not args.no_cuda) and
                             torch.cuda.is_available() else 'cpu')

    return args, job_args, path_args, wandb_args, \
           dynamics_args, queries_args, uncertain_args


def _evaluate_queries(args, job_args, dynamics_args, queries_args,
                      verbose=False):
    # set-up config
    if args.restore_dynamics_from_wandb:

        # get remote config from wandb
        assert args.wandb_dynamics_run_path is not None, \
            'wandb-dynamics-run-path cannot be None'
        run = wandb.Api().run(args.wandb_dynamics_run_path)

        # preserve original dynamics args
        for _arg in dynamics_args._group_actions:
            setattr(args, _arg.dest, run.config[_arg.dest])
            setattr(dynamics_args, _arg.dest, run.config[_arg.dest])

        # create config
        config = BaseConfig(args, dynamics_args)

        # download dynamics
        root = os.path.dirname(config.checkpoint_path)
        name = os.path.basename(config.checkpoint_path)
        os.makedirs(root, exist_ok=True)
        wandb.restore(name=name, run_path=args.wandb_dynamics_run_path,
                      replace=True, root=root)
    else:
        config = BaseConfig(args, dynamics_args)

    # setup experiment tracking
    if args.use_wandb:
        wandb.init(job_type=args.job,
                   project=args.wandb_project_name + '-' + args.job,
                   settings=wandb.Settings(start_method="thread"),
                   save_code=True)
        wandb.config.update({x.dest: vars(args)[x.dest]
                             for x in job_args._group_actions})
        wandb.config.update({x.dest: vars(args)[x.dest]
                             for x in dynamics_args._group_actions})
        wandb.config.update({x.dest: vars(args)[x.dest]
                             for x in queries_args._group_actions})

    # dynamics setup
    assert os.path.exists(config.checkpoint_path), \
        'dynamics network not found: {}'.format(config.checkpoint_path)
    network = config.get_uniform_dynamics_network()
    state_dict = torch.load(config.checkpoint_path, torch.device('cpu'))
    print('state check-point update:{}'.format(state_dict['update']))
    network.load_state_dict(state_dict['network'])

    # set clipping flags
    if config.args.clip_obs:
        network.enable_obs_clip()
    if config.args.clip_reward:
        network.enable_reward_clip()

    network.eval()
    network = network.to(config.args.device)

    # query-evaluation
    queries = opcc.get_queries(args.env_name)
    predicted_df = evaluate_queries(env_name=args.env_name,
                                    queries=queries,
                                    network=network,
                                    runs=args.eval_runs,
                                    batch_size=args.eval_batch_size,
                                    device=args.device,
                                    mixture=args.mixture,
                                    verbose=verbose)
    query_eval_path = config.evaluate_queries_path(args, queries_args)
    predicted_df.to_pickle(query_eval_path)

    # log data on wandb
    if args.use_wandb:
        wandb.run.summary["model-check-point"] = state_dict['update']

        table = wandb.Table(dataframe=predicted_df)
        wandb.log({'query-eval-data': table})
        wandb.log({"mean/q-value-comparison-a":
                       scatter(table, x="pred_a_mean", y="return_a",
                               title="q-value-comparison-a")})
        wandb.log({"mean/q-value-comparison-b":
                       scatter(table, x="pred_b_mean", y="return_b",
                               title="q-value-comparison-b")})


def _train_dynamics(args, job_args, dynamics_args):
    # training resumption
    if args.resume is not None:
        if args.resume == 'wandb':
            # config restoration
            run = wandb.Api().run(args.wandb_run_path)
            for _arg in job_args._group_actions:
                setattr(args, _arg.dest, run.config[_arg.dest])
                setattr(job_args, _arg.dest, run.config[_arg.dest])
            for _arg in dynamics_args._group_actions:
                setattr(args, _arg.dest, run.config[_arg.dest])
                setattr(dynamics_args, _arg.dest, run.config[_arg.dest])
            config = BaseConfig(args, dynamics_args)

            # download checkpoint
            root = os.path.dirname(config.checkpoint_path)
            name = os.path.basename(config.checkpoint_path)
            os.makedirs(root, exist_ok=True)
            wandb.restore(name=name,
                          run_path=args.wandb_run_path,
                          replace=True,
                          root=root)
        elif args.resume == 'local':
            config = BaseConfig(args, dynamics_args)
            assert os.path.exists(config.checkpoint_path), \
                'no checkpoint found  @ {}'.format(config.checkpoint_path)
        else:
            raise ValueError('invalid value for --resume')
    else:
        config = BaseConfig(args, dynamics_args)

    # enable wandb for experiment tracking
    if args.use_wandb:
        wandb.init(job_type=args.job,
                   # dir=args.wandb_dir,
                   project=args.wandb_project_name + '-' + args.job,
                   settings=wandb.Settings(start_method="thread"),
                   id=(args.wandb_run_path.split('/')[-1]
                       if args.resume is not None else None),
                   resume=(True if args.resume is not None else False),
                   save_code=True)
        if args.resume is None or args.resume == 'local':
            wandb.config.update({x.dest: vars(args)[x.dest]
                                 for x in job_args._group_actions})
            wandb.config.update({x.dest: vars(args)[x.dest]
                                 for x in dynamics_args._group_actions})

    train_dynamics(config)

    if args.use_wandb:
        wandb.finish()


def sr_coverages(loss, confidences, tau_interval=0.01):
    """
    Selective-risk Coverage
    """
    # list of tuples (coverage, selective-risk, tau)
    coverage_sr_tau = []
    for tau in np.arange(0, 1 + 2 * tau_interval, tau_interval):
        non_abstain_filter = confidences >= tau
        if any(non_abstain_filter):
            selective_risk = np.sum(loss[non_abstain_filter])
            selective_risk /= np.sum(non_abstain_filter)
            coverage = np.mean(non_abstain_filter)
            coverage_sr_tau.append((coverage, selective_risk, tau))
        else:
            # 0 risk for 0 coverage
            coverage_sr_tau.append((0, 0, tau))

    coverages, selective_risks, taus = list(zip(*sorted(coverage_sr_tau)))

    assert selective_risks[0] == 0 and coverages[0] == 0
    assert coverages[-1] == 1

    return selective_risks, coverages, taus


def area_under_rcc(selective_risks, coverages):
    assert (0, 0) in list(zip(selective_risks, coverages))
    assert 1 in coverages

    # AURCC ( Area Under Risk-Coverage Curve)
    aurcc = metrics.auc(x=coverages, y=selective_risks)
    return aurcc


def reverse_pair_proportion(loss, confidences):
    # Reverse-pair-proportion
    rpp = np.logical_and(np.expand_dims(loss, 1)
                         < np.expand_dims(loss, 1).transpose(),
                         np.expand_dims(confidences, 1)
                         < np.expand_dims(confidences, 1).transpose()
                         ).mean()
    return rpp


def coverage_resolution(coverages, k):
    # Coverage Resolution (cr_k) : Ideally, we would like it to be 1
    bins = [_ for _ in np.arange(0, 1, 1 / k)]
    cr_k = np.unique(np.digitize(coverages, bins)).size / len(bins)

    return cr_k


def _uncertainty_test(args, job_args, dynamics_args, queries_args,
                      uncertainty_args):
    # restore query evaluation data
    if args.restore_query_eval_data_from_wandb:
        assert args.wandb_query_eval_data_run_path is not None, \
            'wandb-query-eval-data-run-path cannot be None'
        run = wandb.Api().run(args.wandb_query_eval_data_run_path)
        remote_config = run.config

        # preserve original dynamics args
        for _arg in dynamics_args._group_actions:
            setattr(args, _arg.dest, remote_config[_arg.dest])
            setattr(dynamics_args, _arg.dest, remote_config[_arg.dest])

        # preserve original query eval args
        for _arg in queries_args._group_actions:
            setattr(args, _arg.dest, remote_config[_arg.dest])
            setattr(queries_args, _arg.dest, remote_config[_arg.dest])

        # download query-evaluation data
        table_file_path = run.summary.get('query-eval-data').get("path")
        table_file = wandb.restore(table_file_path,
                                   args.wandb_query_eval_data_run_path)
        table_str = table_file.read()
        table_dict = json.loads(table_str)

        config = BaseConfig(args, dynamics_args)
        query_eval_df = pd.DataFrame(**table_dict)
    else:
        config = BaseConfig(args, dynamics_args)
        query_eval_path = config.evaluate_queries_path(args, queries_args)
        query_eval_df = pd.read_pickle(query_eval_path)

    # enable wandb for experiment tracking
    if args.use_wandb:
        wandb.init(job_type=args.job,
                   project=args.wandb_project_name + '-' + args.job,
                   settings=wandb.Settings(start_method="thread"),
                   save_code=True)
        wandb.config.update({x.dest: vars(args)[x.dest]
                             for x in job_args._group_actions})
        wandb.config.update({x.dest: vars(args)[x.dest]
                             for x in dynamics_args._group_actions})
        wandb.config.update({x.dest: vars(args)[x.dest]
                             for x in queries_args._group_actions})
        wandb.config.update({x.dest: vars(args)[x.dest]
                             for x in uncertainty_args._group_actions})

    # ################
    # uncertainty-test
    # ################
    query_eval_df['query_distance_a'] = query_eval_df[
        'distance_a-{}'.format(args.dataset_name)]
    query_eval_df['query_distance_b'] = query_eval_df[
        'distance_b-{}'.format(args.dataset_name)]

    if config.args.uncertainty_test_type == 'ensemble-voting':
        uncertainty_dict = ev(query_eval_df,
                              ensemble_size_interval=10,
                              num_ensemble=config.args.num_ensemble)
    elif config.args.uncertainty_test_type == 'paired-confidence-interval':
        uncertainty_dict = ci(query_eval_df,
                              ensemble_size_interval=10,
                              num_ensemble=config.args.num_ensemble,
                              paired=True)
    elif config.args.uncertainty_test_type == 'unpaired-confidence-interval':
        uncertainty_dict = ci(query_eval_df,
                              ensemble_size_interval=10,
                              num_ensemble=config.args.num_ensemble,
                              paired=False)
    else:
        raise NotImplementedError(
            '{} is not implemented'.format(config.args.uncetainty_test))

    # setup logging
    exp_dir = config.uncertainty_exp_dir(args, queries_args, uncertainty_args)
    logs_dir = os.path.join(exp_dir, 'uncertainty_logs')
    os.makedirs(logs_dir, exist_ok=True)
    init_logger(logs_dir, 'uncertainty_test')

    # ###################
    # Evaluation Metrics
    # ###################
    k = 10
    eval_metric_df = []
    sr_coverages_data = {}
    for ensemble_count in sorted(uncertainty_dict.keys()):
        horizons = sorted(uncertainty_dict[ensemble_count].keys())
        sr_coverages_data[ensemble_count] = {}

        for horizon in (horizons + [None]):
            if horizon is None:  # get data for all horizons
                v = defaultdict(lambda: [])
                for _h in horizons:
                    for __k in uncertainty_dict[ensemble_count][_h]:
                        v[__k] += list(uncertainty_dict[ensemble_count]
                                       [_h][__k])

            else:  # get data for specific horizon
                v = uncertainty_dict[ensemble_count][horizon]

            # we use 0-1 loss
            mean_distance = (np.array(v['query_distance_a']) +
                             np.array(v['query_distance_b'])) / 2
            _log = {}
            misc_log = {}
            for distances in [(None, None)]:
                _left, _right = distances
                aurcc_key, rpp_key, cr_k_key = "", "", ""
                if _left is not None and _right is not None:
                    distance_filter = np.logical_and((_left <= mean_distance),
                                                     mean_distance < _right)
                    aurcc_key = "{}_aurcc_{}".format(_left, _right)
                    rpp_key = "{}_rpp_{}".format(_left, _right)
                    cr_k_key = "{}_cr_k_{}".format(_left, _right)
                    loss_key = "{}_loss_{}".format(_left, _right)
                    risk_key = "{}_risk_{}".format(_left, _right)
                    coverage_key = "{}_coverage_{}".format(_left, _right)
                    confidence_key = "{}_confidence_{}".format(_left, _right)
                    taus_key = "{}_taus_{}".format(_left, _right)

                elif _left is not None and _right is None:
                    distance_filter = (_left <= mean_distance)
                    aurcc_key = "{}_aurcc".format(_left)
                    rpp_key = "{}_rpp".format(_left)
                    cr_k_key = "{}_cr_k".format(_left)
                    loss_key = "{}_loss".format(_left)
                    risk_key = "{}_risk".format(_left)
                    coverage_key = "{}_coverage".format(_left)
                    confidence_key = "{}_confidence".format(_left)
                    taus_key = "{}_taus".format(_left)

                elif _left is None and _right is None:
                    distance_filter = np.ones_like(mean_distance).astype(bool)
                    aurcc_key = "aurcc"
                    rpp_key = "rpp"
                    cr_k_key = "cr_k"
                    loss_key = "loss"
                    risk_key = "risk"
                    coverage_key = "coverage"
                    confidence_key = "confidence"
                    taus_key = "taus"
                    distance_key = "distance"
                else:
                    raise NotImplementedError

                loss = np.logical_xor(
                    np.array(v['prediction'])[distance_filter],
                    np.array(v['target'])[distance_filter])
                # evaluation metrics
                selective_risks, coverages, \
                taus = sr_coverages(loss,
                                    np.array(v['confidence'])[distance_filter])
                aurcc = area_under_rcc(selective_risks, coverages)
                rpp = reverse_pair_proportion(loss,
                                              np.array(v['confidence'])[
                                                  distance_filter])
                cr_k = coverage_resolution(coverages, k)

                # log-metrics
                _log = {**_log,
                        **{aurcc_key: aurcc,
                           rpp_key: rpp,
                           cr_k_key: cr_k,
                           loss_key: loss,
                           confidence_key: np.array(v['confidence'])[
                               distance_filter],
                           distance_key: mean_distance[distance_filter]}
                        }
                misc_log = {**misc_log, **{
                    risk_key: selective_risks,
                    coverage_key: coverages,
                    taus_key: taus,
                    distance_key: mean_distance[distance_filter],
                }}

            _log = {**_log,
                    **{'horizon': horizon,
                       'ensemble_count': ensemble_count}}

            logging.getLogger('uncertainty_test').info(_log)
            eval_metric_df.append(pd.DataFrame({_k: [_v]
                                                for _k, _v in _log.items()}))

            # log sr-coverage data
            sr_coverages_data[ensemble_count][horizon] = misc_log

    eval_metric_df = pd.concat(eval_metric_df, ignore_index=True)

    # save data
    uncertainty_dict_path = os.path.join(exp_dir, 'uncertainty_dict.pkl')
    eval_metric_df_path = os.path.join(exp_dir, 'eval_metric_df.pkl')
    sr_coverages_data_path = os.path.join(exp_dir, 'sr_coverage_dict.pkl')

    pickle.dump(uncertainty_dict, open(uncertainty_dict_path, 'wb'))
    pickle.dump(sr_coverages_data, open(sr_coverages_data_path, 'wb'))
    eval_metric_df.to_pickle(eval_metric_df_path)

    # save on wandb
    if args.use_wandb:

        # plot sr-coverage curve data
        for ensemble_count in sr_coverages_data:
            for horizon, _data in sr_coverages_data[ensemble_count].items():
                data = [[x, y] for (x, y) in zip(_data['coverage'],
                                                 _data['risk'])]
                table = wandb.Table(data=data, columns=['coverage', 'risk'])
                if horizon is None:
                    horizon = 'all'
                title = ("SR-Coverage Curve (Ensemble: {}, Horizon:{})".
                         format(ensemble_count, horizon))
                wandb.log({"selective_risk-coverage-ensemble-count-{}"
                           "-horizon-{}".format(ensemble_count, horizon):
                               wandb.plot.line(table, "coverage", "risk",
                                               title=title)})
        # for ensemble_count in sr_coverages_data:
        #     _data = sr_coverages_data[ensemble_count][None]
        #
        #     risk, coverage, distances = _data['risk'], _data['coverage'], _data['distance']
        #
        # for ensemble_count in sr_coverages_data:
        #     for horizon, _data in sr_coverages_data[ensemble_count].items():
        #         for risk_key, coverage_key,a,b  in [('0_risk_4', '0_coverage_4',0,4),
        #                                         ('4_risk_6', '4_coverage_6',4,6),
        #                                         ('6_risk', '6_coverage',6,np.inf),
        #                                         ('risk', 'coverage',"","")]:
        #
        #             data = [[x, y] for (x, y) in zip(_data[coverage_key],
        #                                              _data[risk_key])]
        #             table = wandb.Table(data=data, columns=[coverage_key, risk_key])
        #
        #             title = ("SR-Coverage Curve (Ensemble: {}, Horizon:{}, [{},{}))".
        #                      format(ensemble_count, horizon, a,b))
        #             wandb.log({"selective_risk-coverage-ensemble-count-{}"
        #                        "-horizon-{}-({},{})".format(ensemble_count, horizon,a,b ):
        #                            wandb.plot.line(table, coverage_key, risk_key,
        #                                            title=title)})

        inaccuracy_distance = []
        ensemble_count_data = []
        loss_fig = go.Figure()
        aurcc_fig = go.Figure()
        conf_fig = go.Figure()
        inaccurate_query_conf_fig = go.Figure()
        accurate_query_conf_fig = go.Figure()
        _data = eval_metric_df[(eval_metric_df['ensemble_count']
                                == 10)
                               & (np.isnan(
            eval_metric_df['horizon'].values))]
        bins = np.array([x for x in np.arange(0,
                                              int(np.ceil(
                                                  max(_data['distance']
                                                      .values[0]))) + 1,
                                              0.25)])
        bin_query_inds = np.digitize(_data['distance'].values[0], bins,
                                     right=False)
        filled_bins, filled_bin_counts = np.unique(bin_query_inds,
                                                   return_counts=True)

        for ensemble_count in [10, 20, 50, 100]:
            _data = eval_metric_df[(eval_metric_df['ensemble_count']
                                    == ensemble_count)
                                   & (np.isnan(
                eval_metric_df['horizon'].values))]

            plot_data_loss = []
            plot_data_aurcc = []
            plot_data_conf = []
            plot_data_inaccurate_query_conf = []
            plot_data_accurate_query_conf = []
            for bin in filled_bins:
                _loss = _data['loss'].values[0][bin_query_inds == bin]
                _conf = _data['confidence'].values[0][bin_query_inds == bin]
                selective_risks, coverages, taus = sr_coverages(_loss, _conf)
                aurcc = area_under_rcc(selective_risks, coverages)
                plot_data_loss.append(_loss.mean())
                plot_data_aurcc.append(aurcc)
                plot_data_conf.append(_conf.mean())
                if len(_conf[_loss == True]) > 0:
                    plot_data_inaccurate_query_conf. \
                        append([bins[bin], _conf[_loss == True].mean()])
                if len(_conf[_loss == False]) > 0:
                    plot_data_accurate_query_conf \
                        .append([bins[bin], _conf[_loss == False].mean()])
            plot_data_accurate_query_conf = np.array(
                plot_data_accurate_query_conf)
            plot_data_inaccurate_query_conf = np.array(
                plot_data_inaccurate_query_conf)

            loss_fig.add_trace(
                go.Scatter(x=bins[filled_bins], y=plot_data_loss,
                           mode='lines+markers',
                           name="ensemble-{}".
                           format(ensemble_count)))
            aurcc_fig.add_trace(go.Scatter(x=bins[filled_bins],
                                           y=plot_data_aurcc,
                                           mode='lines+markers',
                                           name="ensemble-{}".
                                           format(ensemble_count)))
            conf_fig.add_trace(go.Scatter(x=bins[filled_bins],
                                          y=plot_data_conf,
                                          mode='lines+markers',
                                          name="ensemble-{}".
                                          format(ensemble_count)))
            inaccurate_query_conf_fig.add_trace(
                go.Scatter(x=plot_data_inaccurate_query_conf[:, 0],
                           y=plot_data_inaccurate_query_conf[:, 1],
                           mode='lines+markers',
                           name="ensemble-{}".format(ensemble_count)))
            accurate_query_conf_fig.add_trace(go.Scatter(
                x=plot_data_accurate_query_conf[:, 0],
                y=plot_data_accurate_query_conf[:, 1],
                mode='lines+markers',
                name="ensemble-{}".format(ensemble_count)))

        # Overlay both histograms
        loss_fig.update_layout(xaxis_title_text='query-distance',
                               yaxis_title_text='Loss',
                               title="Query Distance vs Loss")
        aurcc_fig.update_layout(xaxis_title_text='query-distance',
                                # xaxis label
                                yaxis_title_text='AURCC',
                                title="Query Distance vs AURCC")
        conf_fig.update_layout(xaxis_title_text='query-distance',
                               # xaxis label
                               yaxis_title_text='Mean Confidence',
                               title="Query Distance vs Mean confidence")
        inaccurate_query_conf_fig.update_layout(
            xaxis_title_text='query-distance',
            yaxis_title_text='Mean Confidence',
            title='Query Distance vs "Mean confidence of In-Correct Queries"')
        accurate_query_conf_fig.update_layout(
            xaxis_title_text='query-distance',
            yaxis_title_text='Mean Confidence',
            title='Query Distance vs "Mean confidence of Correct Queries"')
        query_count_fig = go.Figure(data=go.Scatter(x=bins[filled_bins],
                                                    y=filled_bin_counts,
                                                    mode='lines+markers'))
        query_count_fig.update_layout(xaxis_title_text='query-distance',
                                      yaxis_title_text='Query Count',
                                      title="Query Distance vs Count")

        wandb.log({"loss-vs-distance ": loss_fig,
                   "query-count-vs-distance": query_count_fig,
                   "aurcc-vs-distance": aurcc_fig,
                   "confidence-vs-distance": conf_fig,
                   "inaccurate-query-confidence-vs-distance":
                       inaccurate_query_conf_fig,
                   "accurate-query-confidence-vs-distance":
                       accurate_query_conf_fig})
        # import pdb; pdb.set_trace()
        del eval_metric_df['loss']
        del eval_metric_df['distance']
        del eval_metric_df['confidence']
        table = wandb.Table(dataframe=eval_metric_df)
        wandb.log({'eval-metrics': table})
        wandb.save(glob_str=uncertainty_dict_path, policy='now')
        wandb.save(glob_str=sr_coverages_data_path, policy='now')
        wandb.finish()


def main():
    (args, job_args, path_args, wandb_args,
     dynamics_args, queries_args, uncertainty_args) = get_args()
    try:
        if args.job == 'train-dynamics':
            _train_dynamics(args, job_args, dynamics_args)
        elif args.job == 'evaluate-queries':
            _evaluate_queries(args, job_args, dynamics_args, queries_args,
                              verbose=True)
        elif args.job == 'uncertainty-test':
            _uncertainty_test(args, job_args, dynamics_args, queries_args,
                              uncertainty_args)
        else:
            raise NotImplementedError('{} job is not'
                                      ' implemented'.format(args.job))
    except Exception as ex:
        from core.utils import log_traceback
        print("\n".join(log_traceback(ex)))
        if args.use_wandb:
            wandb.finish(exit_code=1)


if __name__ == '__main__':
    main()
