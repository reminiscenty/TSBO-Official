import argparse
import gc
import logging
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from tqdm.auto import tqdm, trange
from botorch.utils.transforms import unnormalize, normalize
from gpytorch.utils.errors import NotPSDError
import copy
ROOT_PROJECT = str(Path(os.path.realpath(__file__)).parent.parent.parent.parent)
if os.path.join(ROOT_PROJECT, 'weighted_retraining') in sys.path:
    sys.path.remove(os.path.join(ROOT_PROJECT, 'weighted_retraining'))
sys.path[0] = ROOT_PROJECT

from weighted_retraining.weighted_retraining.utils import SubmissivePlProgressbar, DataWeighter, print_flush

from weighted_retraining.weighted_retraining.bo_torch.gp_torch import bo_loop, add_gp_torch_args, gp_torch_train, \
    gp_fit_test
from weighted_retraining.test.test_stationarity import make_local_stationarity_plots
from weighted_retraining.weighted_retraining.bo_torch.utils import put_max_in_bounds
from weighted_retraining.weighted_retraining.metrics import METRIC_LOSSES
from teacher_student_scripts.tsbo_train import setup_args, train_teacher_student, add_tsbo_args, add_tsparams_to_path, get_bounds
from torch.utils.data import TensorDataset, Dataset, DataLoader
from utils.utils_cmd import parse_list, parse_dict
from utils.utils_save import get_storage_root, save_w_pickle, str_dict
from weighted_retraining.weighted_retraining.metrics import METRIC_LOSSES
from weighted_retraining.weighted_retraining.robust_opt_scripts.base import add_common_args

# My imports
from weighted_retraining.weighted_retraining import GP_TRAIN_FILE, GP_OPT_FILE
from weighted_retraining.weighted_retraining.chem.chem_data import (
    WeightedJTNNDataset,
    WeightedMolTreeFolder,
    get_rec_x_error)
from weighted_retraining.weighted_retraining.chem.jtnn.datautils import tensorize
from weighted_retraining.weighted_retraining.chem.chem_model import JTVAE
from weighted_retraining.weighted_retraining import utils
from weighted_retraining.weighted_retraining.chem.chem_utils import rdkit_quiet, standardize_smiles
from weighted_retraining.weighted_retraining.robust_opt_scripts import base as wr_base
from weighted_retraining.weighted_retraining.turbo.turbo_1 import Turbo1

logger = logging.getLogger("chem-opt")


def setup_logger(logfile):
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler(logfile)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)


def _run_command(command, command_name):
    logger.debug(f"{command_name} command:")
    logger.debug(command)
    start_time = time.time()
    run_result = subprocess.run(command, capture_output=True)
    print(command)
    print(run_result.stderr)
    print(run_result.stdout)
    # assert run_result.returncode == 0, run_result.stderr
    logger.debug(f"{command_name} done in {time.time() - start_time:.1f}s")


def _batch_decode_z_and_props(
        model: JTVAE,
        z: torch.Tensor,
        datamodule: WeightedJTNNDataset,
        invalid_score: float,
        pbar: tqdm = None,
        evaluation = True
):
    """
    helper function to decode some latent vectors and calculate their properties
    """

    # Progress bar description
    if pbar is not None:
        old_desc = pbar.desc
        pbar.set_description("decoding")

    # Decode all points in a fixed decoding radius
    z_decode = []
    batch_size = 1
    for j in range(0, len(z), batch_size):
        with torch.no_grad():
            z_batch = z[j: j + batch_size]
            smiles_out = model.decode_deterministic(z_batch)
            if pbar is not None:
                pbar.update(z_batch.shape[0])
        z_decode += smiles_out

    # Now finding properties
    if pbar is not None:
        pbar.set_description("calc prop")

    # Calculate objective function values and choose which points to keep
    # Invalid points get a value of None
    if evaluation:
        z_prop = [
            invalid_score if s is None else datamodule.train_dataset.prop_func(s)
            for s in z_decode
        ]
    else:
        z_prop = []

    # Now back to normal
    if pbar is not None:
        pbar.set_description(old_desc)

    return z_decode, z_prop


def _choose_best_rand_points(n_rand_points: int, n_best_points: int, dataset: WeightedMolTreeFolder):
    chosen_point_set = set()

    if len(dataset.data) < n_best_points + n_rand_points:
        n_best_points, n_rand_points = int(n_best_points / (n_best_points + n_rand_points) * len(dataset.data)), int(
            n_rand_points / (n_best_points + n_rand_points) * len(dataset.data))
        n_rand_points += 1 if n_best_points + n_rand_points < len(dataset.data) else 0
    print(f"Take {n_best_points} best points and {n_rand_points} random points")

    # Best scores at start
    targets_argsort = np.argsort(-dataset.data_properties.flatten())
    for i in range(n_best_points):
        chosen_point_set.add(targets_argsort[i])
    candidate_rand_points = np.random.choice(
        len(targets_argsort),
        size=n_rand_points + n_best_points,
        replace=False,
    )
    for i in candidate_rand_points:
        if i not in chosen_point_set and len(chosen_point_set) < (n_rand_points + n_best_points):
            chosen_point_set.add(i)
    assert len(chosen_point_set) == (n_rand_points + n_best_points)
    chosen_points = sorted(list(chosen_point_set))

    return chosen_points


def _encode_mol_trees(model, mol_trees):
    batch_size = 64
    mu_list = []
    with torch.no_grad():
        for i in trange(
                0, len(mol_trees), batch_size, desc="encoding GP points", leave=False
        ):
            batch_slice = slice(i, i + batch_size)
            _, jtenc_holder, mpn_holder = tensorize(
                mol_trees[batch_slice], model.jtnn_vae.vocab, assm=False
            )
            tree_vecs, _, mol_vecs = model.jtnn_vae.encode(jtenc_holder, mpn_holder)
            muT = model.jtnn_vae.T_mean(tree_vecs)
            muG = model.jtnn_vae.G_mean(mol_vecs)
            mu = torch.cat([muT, muG], axis=-1).cpu().numpy()
            mu_list.append(mu)

    # Aggregate array
    mu = np.concatenate(mu_list, axis=0).astype(np.float32)
    return mu


def retrain_model(model, datamodule, save_dir, version_str, num_epochs, cuda, store_best=False,
                  best_ckpt_path: Optional[str] = None):
    # pl._logger.setLevel(logging.CRITICAL)
    train_pbar = SubmissivePlProgressbar(process_position=1)

    # Create custom saver and logger
    tb_logger = TensorBoardLogger(
        save_dir=save_dir, version=version_str, name=""
    )
    checkpointer = pl.callbacks.ModelCheckpoint(save_last=True, monitor="loss/val", )

    # Handle fractional epochs
    if num_epochs < 1:
        max_epochs = 1
        limit_train_batches = num_epochs
    elif int(num_epochs) == num_epochs:
        max_epochs = int(num_epochs)
        limit_train_batches = 1.0
    else:
        raise ValueError(f"invalid num epochs {num_epochs}")

    # Create trainer
    trainer = pl.Trainer(
        gpus=[cuda] if cuda is not None else 0,
        max_epochs=max_epochs,
        limit_train_batches=limit_train_batches,
        limit_val_batches=1,
        checkpoint_callback=True,
        terminate_on_nan=True,
        logger=tb_logger,
        callbacks=[train_pbar, checkpointer],
        gradient_clip_val=20.0,  # Model is prone to large gradients
    )

    # Fit model
    try:
        trainer.fit(model, datamodule)
    except:
        print('Fail to finish retraining')

    if store_best:
        assert best_ckpt_path is not None
        os.makedirs(os.path.dirname(best_ckpt_path), exist_ok=True)
        shutil.copyfile(checkpointer.best_model_path, best_ckpt_path)


def get_root_path(lso_strategy: str, weight_type, k, r,
                  predict_target, hdims, latent_dim: int, beta_kl_final: float, beta_metric_loss: float,
                  beta_target_pred_loss: float,
                  metric_loss: str, metric_loss_kw: Dict[str, Any],
                  acq_func_id: str, acq_func_kwargs: Dict[str, Any],
                  input_wp: bool,
                  random_search_type: Optional[str],
                  use_pretrained: bool, pretrained_model_id: str, batch_size: int,
                  n_init_retrain_epochs: float, semi_supervised: Optional[bool], n_init_bo_points: Optional[int]
                  ):
    """ Get result root result path (associated directory will contain results for all seeds)
    Args:
        batch_size: batch size used for vae training
        pretrained_model_id: id of the pretrained model
        lso_strategy: type of optimisation
        weight_type: type of weighting used for retraining
        k: weighting parameter
        r: period of retraining
        predict_target: whether generative model also predicts target value
        hdims: latent dims of target MLP predictor
        metric_loss: metric loss used to structure embedding space
        metric_loss_kw: kwargs for metric loss
        acq_func_id: name of acquisition function
        acq_func_kwargs: acquisition function kwargs
        random_search_type: random search specific strategy
        beta_metric_loss: weight of the metric loss added to the ELBO
        beta_kl_final: weight of the KL in the ELBO
        beta_target_pred_loss: weight of the target prediction loss added to the ELBO
        latent_dim: dimension of the latent space
        use_pretrained: Whether or not to use a pretrained VAE model
        n_init_retrain_epochs: number of retraining epochs to do before using VAE model in BO
        semi_supervised: whether or not to start BO from VAE trained with unlabelled data
        n_init_bo_points: number of initial labelled points considered for BO with semi-supervised training

    Returns:
        path to result dir
    """
    result_path = os.path.join(
        get_storage_root(),
        f"logs/opt/chem/{weight_type}/k_{k}/r_{r}")

    exp_spec = f"paper-mol"
    exp_spec += f'-z_dim_{latent_dim}'
    exp_spec += f"-init_{n_init_retrain_epochs:g}"
    if predict_target:
        assert hdims is not None
        exp_spec += '-predy_' + '_'.join(map(str, hdims))
        exp_spec += f'-b_{float(beta_target_pred_loss):g}'
    if metric_loss is not None:
        print(metric_loss, METRIC_LOSSES[metric_loss]['exp_metric_id'](**metric_loss_kw))
        exp_spec += '-' + METRIC_LOSSES[metric_loss]['exp_metric_id'](**metric_loss_kw)
        exp_spec += f'-b_{float(beta_metric_loss):g}'
    exp_spec += f'-bkl_{beta_kl_final}'
    if semi_supervised:
        assert n_init_bo_points is not None, n_init_bo_points
        exp_spec += "-semi_supervised"
        exp_spec += f"-n-init-{n_init_bo_points}"
    if use_pretrained:
        exp_spec += f'_pretrain-{pretrained_model_id}'
    else:
        exp_spec += f'_scratch'
    if batch_size != 32:
        exp_spec += f'_bs-{batch_size}'

    if lso_strategy == 'opt':
        acq_func_spec = ''
        if acq_func_id != 'ExpectedImprovement':
            acq_func_spec += acq_func_id

        acq_func_spec += f"{'_inwp_' if input_wp else str(input_wp)}" \
            # if 'ErrorAware' in acq_func_id and cost_aware_gamma_sched is not None:
        #     acq_func_spec += f"_sch-{cost_aware_gamma_sched}"
        if len(acq_func_kwargs) > 0:
            acq_func_spec += f'_{str_dict(acq_func_kwargs)}'
        result_path = os.path.join(
            result_path, exp_spec, acq_func_spec
        )

    elif lso_strategy == 'sample':
        # raise NotImplementedError('Sample lso strategy not supported')
        result_path = os.path.join(result_path, exp_spec, f'latent-sample')
    elif lso_strategy == 'random_search':
        base = f'latent-random-search'
        if random_search_type == 'sobol':
            base += '-sobol'
        else:
            assert random_search_type is None, f'{random_search_type} is invalid'
        result_path = os.path.join(result_path, exp_spec, base)
    elif lso_strategy =='turbo':
        result_path = os.path.join(result_path, exp_spec, lso_strategy)
    else:
        raise ValueError(f'{lso_strategy} not supported: try `opt`, `sample`...')
        

    return result_path


def get_path(lso_strategy: str, weight_type, k, r,
             predict_target, hdims, latent_dim: int, beta_kl_final: float, beta_metric_loss: float,
             beta_target_pred_loss: float,
             metric_loss: str, metric_loss_kw: Dict[str, Any],
             acq_func_id: str, acq_func_kwargs: Dict[str, Any],
             input_wp: bool,
             random_search_type: Optional[str],
             use_pretrained: bool, pretrained_model_id: str, batch_size: int,
             n_init_retrain_epochs: int, seed: float, semi_supervised: Optional[bool], n_init_bo_points: Optional[int], 
             args):
    """ Get result root result path (associated directory will contain results for all seeds)
    Args:
        batch_size: batch size used for vae training
        pretrained_model_id: id of the pretrained model
        seed: for reproducibility
        lso_strategy: type of optimisation
        weight_type: type of weighting used for retraining
        k: weighting parameter
        r: period of retraining
        predict_target: whether generative model also predicts target value
        hdims: latent dims of target MLP predictor
        metric_loss: metric loss used to structure embedding space
        metric_loss_kw: kwargs for metric loss
        acq_func_id: name of acquisition function
        acq_func_kwargs: acquisition function kwargs
        random_search_type: random search specific strategy
        beta_metric_loss: weight of the metric loss added to the ELBO
        beta_kl_final: weight of the KL in the ELBO
        beta_target_pred_loss: weight of the target prediction loss added to the ELBO
        latent_dim: dimension of the latent space
        use_pretrained: Whether or not to use a pretrained VAE model
        n_init_retrain_epochs: number of retraining epochs to do before using VAE model in BO
        semi_supervised: whether or not to start BO from VAE trained with unlabelled data
        n_init_bo_points: number of initial labelled points considered for BO
    Returns:
        path to result dir
    """
    result_path = get_root_path(
        lso_strategy=lso_strategy,
        weight_type=weight_type,
        k=k,
        r=r,
        predict_target=predict_target,
        latent_dim=latent_dim,
        hdims=hdims,
        metric_loss=metric_loss,
        metric_loss_kw=metric_loss_kw,
        acq_func_id=acq_func_id,
        acq_func_kwargs=acq_func_kwargs,
        input_wp=input_wp,
        random_search_type=random_search_type,
        beta_target_pred_loss=beta_target_pred_loss,
        beta_metric_loss=beta_metric_loss,
        beta_kl_final=beta_kl_final,
        use_pretrained=use_pretrained,
        n_init_retrain_epochs=n_init_retrain_epochs,
        batch_size=batch_size,
        semi_supervised=semi_supervised,
        n_init_bo_points=n_init_bo_points,
        pretrained_model_id=pretrained_model_id
    )
    result_path = add_tsparams_to_path(args,result_path)
    result_path = os.path.join(result_path, f'seed{seed}')
    return result_path


def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.register('type', list, parse_list)
    parser.register('type', dict, parse_dict)

    parser = add_common_args(parser)
    parser = WeightedJTNNDataset.add_model_specific_args(parser)
    parser = DataWeighter.add_weight_args(parser)
    parser = wr_base.add_gp_args(parser)
    parser = add_tsbo_args(parser)

    parser.add_argument(
        "--cuda",
        type=int,
        default=None,
        help="cuda ID",
    )

    parser.add_argument(
        '--use_test_set',
        dest="use_test_set",
        action="store_true",
        help="flag to use a test set for evaluating the sparse GP"
    )
    parser.add_argument(
        '--use_full_data_for_gp',
        dest="use_full_data_for_gp",
        action="store_true",
        help="flag to use the full dataset for training the GP"
    )
    parser.add_argument(
        "--input_wp",
        action='store_true',
        help="Whether to apply input warping"
    )
    parser.add_argument(
        "--predict_target",
        action='store_true',
        help="Generative model predicts target value",
    )
    parser.add_argument(
        "--target_predictor_hdims",
        type=list,
        default=None,
        help="Hidden dimensions of MLP predicting target values",
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=56,
        help="Hidden dimension the latent space",
    )
    parser.add_argument(
        "--use_pretrained",
        action='store_true',
        help="True if using pretrained VAE model",
    )
    parser.add_argument(
        "--pretrained_model_id",
        type=str,
        default='vanilla',
        help="id of the pretrained VAE model used (should be aligned with the pretrained model file)",
    )

    vae_group = parser.add_argument_group("Metric learning")
    vae_group.add_argument(
        "--metric_loss",
        type=str,
        help="Metric loss to add to VAE loss during training of the generative model to get better "
             "structured latent space (see `METRIC_LOSSES`), one of ['contrastive', 'triplet', 'log_ratio', 'infob']",
    )
    vae_group.add_argument(
        "--metric_loss_kw",
        type=dict,
        default=None,
        help="Threshold parameter for contrastive loss, one of [{'threshold':.1}, {'threshold':.1,'margin':1}]",
    )
    vae_group.add_argument(
        "--beta_target_pred_loss",
        type=float,
        default=1.,
        help="Weight of the target_prediction loss added in the ELBO",
    )
    vae_group.add_argument(
        "--beta_metric_loss",
        type=float,
        default=1.,
        help="Weight of the metric loss added in the ELBO",
    )
    vae_group.add_argument(
        "--beta_final",
        type=float,
        help="Weight of the kl loss in the ELBO",
    )
    vae_group.add_argument(
        "--beta_start",
        type=float,
        default=None,
        help="Weight of the kl loss in the ELBO",
    )
    vae_group.add_argument(
        "--semi_supervised",
        action='store_true',
        help="Start BO from VAE trained with unlabelled data.",
    )
    vae_group.add_argument(
        "--n_init_bo_points",
        type=int,
        default=None,
        help="Number of data points to use at the start of the BO if using semi-supervised training of VAE."
             "(We need at least SOME data to fit the GP(s) etc.)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0007,
        help="learning rate of the VAE training optimizer if needed (e.g. in case VAE from scratch)",
    )
    parser.add_argument(
        "--train-only",
        action='store_true',
        help="Train the JTVAE without running the BO",
    )
    parser.add_argument(
        "--save_model_path",
        type=str,
        default=None,
        help="If `train-only`, save the trained model in save_model_path.",
    )
    args = parser.parse_args()

    args.train_path = os.path.join(ROOT_PROJECT, args.train_path)
    args.val_path = os.path.join(ROOT_PROJECT, args.val_path)
    args.vocab_file = os.path.join(ROOT_PROJECT, args.vocab_file)
    args.property_file = os.path.join(ROOT_PROJECT, args.property_file)

    if 'ErrorAware' in args.acq_func_id:
        assert 'gamma' in args.acq_func_kwargs
        assert 'eta' in args.acq_func_kwargs
        args.error_aware_acquisition = True
    else:
        args.error_aware_acquisition = False

    if args.pretrained_model_file is None:
        if args.use_pretrained:
            raise ValueError("You should specify the path to the pretrained model you want to use via "
                             "--pretrained_model_file argument")

    # Seeding
    pl.seed_everything(args.seed)

    setup_args(args,'chem')

    # create result directory
    result_dir = get_path(
        lso_strategy=args.lso_strategy,
        weight_type=args.weight_type,
        k=args.rank_weight_k,
        r=args.retraining_frequency,
        predict_target=args.predict_target,
        latent_dim=args.latent_dim,
        hdims=args.target_predictor_hdims,
        metric_loss=args.metric_loss,
        metric_loss_kw=args.metric_loss_kw,
        input_wp=args.input_wp,
        seed=args.seed,
        random_search_type=args.random_search_type,
        beta_metric_loss=args.beta_metric_loss,
        beta_target_pred_loss=args.beta_target_pred_loss,
        beta_kl_final=args.beta_final,
        use_pretrained=args.use_pretrained,
        n_init_retrain_epochs=args.n_init_retrain_epochs,
        semi_supervised=args.semi_supervised,
        n_init_bo_points=args.n_init_bo_points,
        pretrained_model_id=args.pretrained_model_id,
        batch_size=args.batch_size,
        acq_func_id=args.acq_func_id,
        acq_func_kwargs=args.acq_func_kwargs,
        args=args
    )
    print(f'result dir: {result_dir}')
    os.makedirs(result_dir, exist_ok=True)
    save_w_pickle(args, result_dir, 'args.pkl')
    logs = ''
    exc: Optional[Exception] = None
    try:
        main_aux(args, result_dir=result_dir)
    except Exception as e:
        logs = traceback.format_exc()
        exc = e
    f = open(os.path.join(result_dir, 'logs.txt'), "a")
    f.write('\n' + '--------' * 10)
    f.write(logs)
    f.write('\n' + '--------' * 10)
    f.close()
    if exc is not None:
        raise exc


def main_aux(args, result_dir: str):
    """ main """

    # Seeding
    pl.seed_everything(args.seed)
    print(args.gpu, args.cuda)
    device = args.cuda
    if device is not None:
        torch.cuda.set_device(device)
    tkwargs = {
        "dtype": torch.float,
        "device": torch.device(f"cuda:{device}" if torch.cuda.is_available() and device is not None else "cpu"),
    }

    if args.train_only and os.path.exists(args.save_model_path) and not args.overwrite:
        print_flush(f'--- JTVAE already trained in {args.save_model_path} ---')
        return

    # Make results directory
    data_dir = os.path.join(result_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    setup_logger(os.path.join(result_dir, "log.txt"))

    # Load data
    datamodule = WeightedJTNNDataset(args, utils.DataWeighter(args))
    datamodule.setup("fit", n_init_points=args.n_init_bo_points)

    # add additional noise
    if args.is_additional_noise:
        datamodule.train_dataset.data_properties += args.additional_noise[:len(datamodule.train_dataset.data_properties)]

    # print python command run
    cmd = ' '.join(sys.argv[1:])
    print_flush(f"{cmd}\n")

    # Load model
    if args.use_pretrained:
        if args.predict_target:
            if 'pred_y' in args.pretrained_model_file:
                # fully supervised training from a model already trained with target prediction
                ckpt = torch.load(args.pretrained_model_file)
                ckpt['hyper_parameters']['hparams'].beta_target_pred_loss = args.beta_target_pred_loss
                ckpt['hyper_parameters']['hparams'].predict_target = True
                ckpt['hyper_parameters']['hparams'].target_predictor_hdims = args.target_predictor_hdims
                torch.save(ckpt, args.pretrained_model_file)
        print(os.path.abspath(args.pretrained_model_file))
        vae: JTVAE = JTVAE.load_from_checkpoint(args.pretrained_model_file, vocab=datamodule.vocab)
        vae.beta = vae.hparams.beta_final  # Override any beta annealing
        vae.metric_loss = args.metric_loss
        vae.hparams.metric_loss = args.metric_loss
        vae.beta_metric_loss = args.beta_metric_loss
        vae.hparams.beta_metric_loss = args.beta_metric_loss
        vae.metric_loss_kw = args.metric_loss_kw
        vae.hparams.metric_loss_kw = args.metric_loss_kw
        vae.predict_target = args.predict_target
        vae.hparams.predict_target = args.predict_target
        vae.beta_target_pred_loss = args.beta_target_pred_loss
        vae.hparams.beta_target_pred_loss = args.beta_target_pred_loss
        vae.target_predictor_hdims = args.target_predictor_hdims
        vae.hparams.target_predictor_hdims = args.target_predictor_hdims
        if vae.predict_target and vae.target_predictor is None:
            vae.hparams.target_predictor_hdims = args.target_predictor_hdims
            vae.hparams.predict_target = args.predict_target
            vae.build_target_predictor()
    else:
        print("initialising VAE from scratch !")
        vae: JTVAE = JTVAE(hparams=args, vocab=datamodule.vocab)
    vae.eval()

    # Set up some stuff for the progress bar
    num_retrain = int(np.ceil(args.query_budget / args.retraining_frequency))
    postfix = dict(
        retrain_left=num_retrain,
        best=float(datamodule.train_dataset.data_properties.max()),
        n_train=len(datamodule.train_dataset.data),
        save_path=result_dir
    )

    start_num_retrain = 0

    # Set up results tracking
    results = dict(
        opt_points=[],
        opt_point_properties=[],
        opt_model_version=[],
        params=str(sys.argv),
        sample_points=[],
        sample_versions=[],
        sample_properties=[],
    )

    result_filepath = os.path.join(result_dir, 'results.npz')
    if not args.overwrite and os.path.exists(result_filepath):
        with np.load(result_filepath, allow_pickle=True) as npz:
            results = {}
            for k in list(npz.keys()):
                results[k] = npz[k]
                if k != 'params':
                    results[k] = list(results[k])
                else:
                    results[k] = npz[k].item()
        start_num_retrain = results['opt_model_version'][-1] + 1

        prev_retrain_model = args.retraining_frequency * (start_num_retrain - 1)
        num_sampled_points = len(results['opt_points'])
        if args.n_init_retrain_epochs == 0 and prev_retrain_model == 0:
            pretrained_model_path = args.pretrained_model_file
        else:
            pretrained_model_path = os.path.join(result_dir, 'retraining', f'retrain_{prev_retrain_model}',
                                                 'checkpoints',
                                                 'last.ckpt')
        print(f"Found checkpoint at {pretrained_model_path}")
        ckpt = torch.load(pretrained_model_path)
        ckpt['hyper_parameters']['hparams'].metric_loss = args.metric_loss
        ckpt['hyper_parameters']['hparams'].metric_loss_kw = args.metric_loss_kw
        ckpt['hyper_parameters']['hparams'].beta_metric_loss = args.beta_metric_loss
        ckpt['hyper_parameters']['hparams'].beta_target_pred_loss = args.beta_target_pred_loss
        if args.predict_target:
            ckpt['hyper_parameters']['hparams'].predict_target = True
            ckpt['hyper_parameters']['hparams'].target_predictor_hdims = args.target_predictor_hdims
        torch.save(ckpt, pretrained_model_path)
        print(f"Loading model from {pretrained_model_path}")
        vae.load_from_checkpoint(pretrained_model_path, vocab=datamodule.vocab)
        if args.predict_target and not hasattr(vae.hparams, 'predict_target'):
            vae.hparams.target_predictor_hdims = args.target_predictor_hdims
            vae.hparams.predict_target = args.predict_target
        # vae.hparams.cuda = args.cuda
        vae.beta = vae.hparams.beta_final  # Override any beta annealing
        vae.eval()

        # Set up some stuff for the progress bar
        num_retrain = int(np.ceil(args.query_budget / args.retraining_frequency)) - start_num_retrain

        print(f"Append existing points and properties to datamodule...")
        datamodule.append_train_data(
            np.array(results['opt_points']),
            np.array(results['opt_point_properties'])
        )
        postfix = dict(
            retrain_left=num_retrain,
            best=float(datamodule.train_dataset.data_properties.max()),
            n_train=len(datamodule.train_dataset.data),
            initial=num_sampled_points,
            save_path=result_dir
        )
        print(f"Retrain from {result_dir} | Best: {max(results['opt_point_properties'])}")
    start_time = time.time()

    # Main loop
    args.seen_list = []
    with tqdm(
            total=args.query_budget, dynamic_ncols=True, smoothing=0.0, file=sys.stdout
    ) as pbar:

        for ret_idx in range(start_num_retrain, start_num_retrain + num_retrain):

            if vae.predict_target and vae.metric_loss is not None:
                vae.training_m = datamodule.training_m
                vae.training_M = datamodule.training_M
                vae.validation_m = datamodule.validation_m
                vae.validation_M = datamodule.validation_M

            torch.cuda.empty_cache()  # Free the memory up for tensorflow
            pbar.set_postfix(postfix)
            pbar.set_description("retraining")
            print(result_dir)
            # Decide whether to retrain
            samples_so_far = args.retraining_frequency * ret_idx

            # Optionally do retraining
            num_epochs = args.n_retrain_epochs
            if ret_idx == 0 and args.n_init_retrain_epochs is not None:
                num_epochs = args.n_init_retrain_epochs
            if num_epochs > 0:
                retrain_dir = os.path.join(result_dir, "retraining")
                version = f"retrain_{samples_so_far}"
                args.retrain_dir = retrain_dir
                args.version = version
                retrain_model(
                    model=vae, datamodule=datamodule, save_dir=retrain_dir,
                    version_str=version, num_epochs=num_epochs, cuda=args.cuda, store_best=args.train_only,
                    best_ckpt_path=args.save_model_path
                )
                vae.eval()
                print('retrain teacher student')
                args.ts_train_count = 0
                if args.train_only:
                    return
            del num_epochs

            model = vae

            # Update progress bar
            postfix["retrain_left"] -= 1
            pbar.set_postfix(postfix)

            # Draw samples for logs!
            if args.samples_per_model > 0:
                pbar.set_description("sampling")
                with trange(
                        args.samples_per_model, desc="sampling", leave=False
                ) as sample_pbar:
                    sample_x, sample_y = latent_sampling(
                        args, model, datamodule, args.samples_per_model,
                        pbar=sample_pbar
                    )

                # Append to results dict
                results["sample_points"].append(sample_x)
                results["sample_properties"].append(sample_y)
                results["sample_versions"].append(ret_idx)

            # Do querying!
            pbar.set_description("querying")
            num_queries_to_do = min(
                args.retraining_frequency, args.query_budget - samples_so_far
            )
            if args.lso_strategy == "opt":
                gp_dir = os.path.join(result_dir, "gp", f"iter{samples_so_far}")
                os.makedirs(gp_dir, exist_ok=True)
                gp_data_file = os.path.join(gp_dir, "data.npz")
                gp_err_data_file = os.path.join(gp_dir, "data_err.npz")
                x_new, y_new = latent_optimization(
                    args=args,
                    model=model,
                    datamodule=datamodule,
                    n_inducing_points=args.n_inducing_points,
                    n_best_points=args.n_best_points,
                    n_rand_points=args.n_rand_points,
                    tkwargs=tkwargs,
                    num_queries_to_do=num_queries_to_do,
                    gp_data_file=gp_data_file,
                    gp_err_data_file=gp_err_data_file,
                    gp_run_folder=gp_dir,
                    gpu=args.gpu,
                    invalid_score=args.invalid_score,
                    pbar=pbar,
                    postfix=postfix,
                    error_aware_acquisition=args.error_aware_acquisition,
                )
            elif args.lso_strategy == "sample":
                x_new, y_new = latent_sampling(
                    args, model, datamodule, num_queries_to_do, pbar=pbar,
                )
            elif args.lso_strategy == "turbo":
                x_new, y_new = latent_turbo(
                    model=model.to(**tkwargs), 
                    args=args, 
                    tkwargs=tkwargs, 
                    datamodule=datamodule, 
                    num_queries_to_do=num_queries_to_do, 
                )
            else:
                raise NotImplementedError(args.lso_strategy)

            # add additional noise
            # Update dataset
            if args.is_additional_noise:
                datamodule.append_train_data(x_new, y_new+args.additional_noise[len(datamodule.train_dataset.data_properties)])
            else:
                datamodule.append_train_data(x_new, y_new)

            # Add new results
            results["opt_points"] += list(x_new)
            results["opt_point_properties"] += list(y_new)
            results["opt_model_version"] += [ret_idx] * len(x_new)

            postfix["best"] = max(postfix["best"], float(max(y_new)))
            postfix["n_train"] = len(datamodule.train_dataset.data)
            pbar.set_postfix(postfix)

            # Save results
            np.savez_compressed(os.path.join(result_dir, "results.npz"), **results)

            # Keep a record of the dataset here
            new_data_file = os.path.join(
                data_dir, f"train_data_iter{samples_so_far + num_queries_to_do}.txt"
            )
            with open(new_data_file, "w") as f:
                f.write("\n".join(datamodule.train_dataset.canonic_smiles))
    # save to an another folder
    dir = result_dir.split('/')
    new_dir = os.path.join(ROOT_PROJECT,"results/chem",dir[-2],dir[-1])
    os.makedirs(new_dir, exist_ok=True)
    if args.query_budget != 250:
        np.savez_compressed(os.path.join(new_dir, f"results_{args.query_budget}.npz"), **results)
    np.savez_compressed(os.path.join(new_dir, "results.npz"), **results)
    print_flush("=== DONE ({:.3f}s) ===".format(time.time() - start_time))


def get_latent_dataset(model, n_rand_points, n_best_points, dset, tkwargs):
    chosen_indices = _choose_best_rand_points(n_rand_points=n_rand_points, n_best_points=n_best_points, dataset=dset)
    mol_trees = [dset.data[i] for i in chosen_indices]
    targets = dset.data_properties[chosen_indices]
    chosen_smiles = [dset.canonic_smiles[i] for i in chosen_indices]

    # Next, encode these mol trees
    latent_points = _encode_mol_trees(model, mol_trees)
    # do not standardize -> we'll normalize in unit cube
    X_train = torch.tensor(latent_points).to(**tkwargs)
    y_train = torch.tensor(targets).view(-1,1).to(**tkwargs)
    print(X_train.shape, y_train.shape)
    torch.cuda.empty_cache()  # Free the memory up for tensorflow
    return X_train, y_train, latent_points, targets, chosen_smiles
def latent_optimization(
        args,
        model: JTVAE,
        datamodule: WeightedJTNNDataset,
        n_inducing_points: int,
        n_best_points: int,
        n_rand_points: int,
        tkwargs: Dict[str, Any],
        num_queries_to_do: int,
        invalid_score: float,
        gp_data_file: str,
        gp_run_folder: str,
        gpu: bool,
        error_aware_acquisition: bool,
        gp_err_data_file: Optional[str],
        pbar=None,
        postfix=None,
):
    ##################################################
    # Prepare GP
    ##################################################

    # First, choose GP points to train!
    dset = datamodule.train_dataset
    q = 1
    model.to(**tkwargs)
    X_train, y_train, latent_points, targets, chosen_smiles = get_latent_dataset(model, n_rand_points, n_best_points, dset, tkwargs)
    bounds, ybounds = get_bounds(X_train, y_train, tkwargs)
    # model.cpu()  # Make sure to free up GPU memory
    can_smiles_set = set(dset.canonic_smiles)
    seen_list = args.seen_list

    

    def check_whether_seen(s):
        seen = False
        if s is None:
            print('None smile')
            return True
        if s in all_new_smiles:
            print('seen in all_new_smiles')
            return True
        if s in args.seen_list:
            print('seen in seen_list')
            return True
        s_std = standardize_smiles(s)
        # print(s, s_std)
        if s_std in can_smiles_set:
            print('seen in can_smiles_set')
            args.seen_list.append(s)
            return True
        return seen

    # Save points to file
    def _save_gp_data(x, y, s, file, flip_sign=True):

        # Prevent overfitting to bad points
        y = np.maximum(y, invalid_score)
        if flip_sign:
            y = -y.reshape(-1, 1)  # Since it is a maximization problem
        else:
            y = y.reshape(-1, 1)

        # Save the file
        np.savez_compressed(
            file,
            X_train=x.astype(np.float32),
            X_test=[],
            y_train=y.astype(np.float32),
            y_test=[],
            smiles=s,
        )

    # If using error-aware acquisition, compute reconstruction error of selected points
    if error_aware_acquisition:
        assert gp_err_data_file is not None, "Please provide a data file for the error GP"
        if gpu:
            model = model.cuda()
        error_train, safe_idx = get_rec_x_error(
            model,
            tkwargs={'dtype': torch.float},
            data=[datamodule.train_dataset.data[i] for i in chosen_indices],
        )
        # exclude points for which we could not compute the reconstruction error from the objective GP dataset
        if len(safe_idx) < latent_points.shape[0]:
            failed = [i for i in range(latent_points.shape[0]) if i not in safe_idx]
            print_flush(f"Could not compute the recon. err. of {len(failed)} points -> excluding them.")
            latent_points_err = latent_points[safe_idx]
            chosen_smiles_err = [chosen_smiles[i] for i in safe_idx]
        else:
            latent_points_err = latent_points
            chosen_smiles_err = chosen_smiles
        # model = model.cpu()  # Make sure to free up GPU memory
        torch.cuda.empty_cache()  # Free the memory up for tensorflow
        _save_gp_data(latent_points, error_train.cpu().numpy(), chosen_smiles, gp_err_data_file)
    _save_gp_data(latent_points, targets, chosen_smiles, gp_data_file, flip_sign=False)

    ##################################################
    # Run iterative GP fitting/optimization
    ##################################################
    curr_gp_file = None
    curr_gp_err_file = None
    all_new_smiles = []
    all_new_props = []
    all_new_err = []
    rand_point_due_bo_fail = []
    n_rand_acq = 0  # number of times we have to acquire a random point as bo acquisition crashed
    redo_counter = 1
    current_n_inducing_points = min(latent_points.shape[0], n_inducing_points)

    for gp_iter in range(num_queries_to_do):
        gp_initial_train = gp_iter == 0
        if latent_points.shape[0] == n_inducing_points:
            gp_initial_train = True

        # Part 1: fit GP
        # ===============================
        new_gp_file = os.path.join(gp_run_folder, f"gp_train_res{gp_iter:04d}.npz")
        new_gp_error_file = os.path.join(gp_run_folder, f"gp_train_error_res{gp_iter:04d}.npz")
        # log_path = os.path.join(gp_run_folder, f"gp_train{gp_iter:04d}.log")
        iter_seed = int(np.random.randint(10000))

        gp_file = None
        gp_error_file = None
        if gp_iter == 0:
            # Add commands for initial fitting
            gp_fit_desc = "GP initial fit"
            # n_perf_measure = 0
            current_n_inducing_points = min(X_train.shape[0], n_inducing_points)
        else:
            gp_fit_desc = "GP incremental fit"
            gp_file = curr_gp_file
            # n_perf_measure = 1  # specifically see how well it fits the last point!
        init = gp_iter == 0
        # if se;i-supervised training, wait until we have enough points to use as many inducing points as we wanted and re-init GP
        if X_train.shape[0] == n_inducing_points:
            current_n_inducing_points = n_inducing_points
            init = True

        old_desc = None
        # Set pbar status for user
        if pbar is not None:
            old_desc = pbar.desc
            pbar.set_description(gp_fit_desc)

        np.random.seed(iter_seed)
        torch.manual_seed(iter_seed)

        y_mean, y_std = y_train.mean(), y_train.std()
        y_train_step = (y_train - y_mean) / (y_std + 1e-8)

        # To account for outliers
        bounds = torch.zeros(2, X_train.shape[1], **tkwargs)
        bounds[0] = torch.quantile(X_train, .0005, dim=0)
        bounds[1] = torch.quantile(X_train, .9995, dim=0)
        ybounds = torch.zeros(2, y_train.shape[1], **tkwargs)
        ybounds[0] = torch.quantile(y_train, .0005, dim=0)
        ybounds[1] = torch.quantile(y_train, .9995, dim=0)
        ydelta = .05 * (ybounds[1] - ybounds[0])
        ybounds[0] -= ydelta
        ybounds[1] += ydelta

        # make sure best sample is within bounds
        y_train_std = y_train.add(-y_train.mean()).div(y_train.std())
        y_train_normalized = normalize(y_train, ybounds)  # minimize
        bounds = put_max_in_bounds(X_train, y_train_std, bounds)
        # bounds = put_max_in_bounds(X_train, y_train_normalized, bounds)

        # print(f"Data bound of {bounds} found...")
        delta = .05 * (bounds[1] - bounds[0])
        bounds[0] -= delta
        bounds[1] += delta
        # print(f"Using data bound of {bounds}...")

        train_x = normalize(X_train, bounds)
        try:
            if args.self_training:
                print_flush('start training')
                unlabel_x, unlabel_y = train_teacher_student(args, X_train.detach(), y_train.detach(), X_train.detach(), y_train.detach(), bounds, datamodule)
                train_x_arg = torch.cat((train_x, unlabel_x),dim=0)
                y_train_arg = torch.cat((y_train, unlabel_y),dim=0)
                y_train_arg_std = y_train_arg.add(-y_train_arg.mean()).div(y_train_arg.std())
                gp_model = gp_torch_train(
                    train_x=train_x_arg,
                    train_y=y_train_arg_std,
                    # train_y=y_train_normalized,
                    n_inducing_points=current_n_inducing_points,
                    tkwargs=tkwargs,
                    init=init,
                    scale=args.scale,
                    covar_name=args.covar_name,
                    gp_file=gp_file,
                    save_file=new_gp_file,
                    input_wp=args.input_wp,
                    outcome_transform=None,
                    options={'lr': 5e-3, 'maxiter': 500} if init else {'lr': 5e-3, 'maxiter': 100}
                )
            else:
                gp_model = gp_torch_train(
                    train_x=train_x,
                    train_y=y_train_std,
                    # train_y=y_train_normalized,
                    n_inducing_points=current_n_inducing_points,
                    tkwargs=tkwargs,
                    init=init,
                    scale=args.scale,
                    covar_name=args.covar_name,
                    gp_file=gp_file,
                    save_file=new_gp_file,
                    input_wp=args.input_wp,
                    outcome_transform=None,
                    options={'lr': 5e-3, 'maxiter': 500} if init else {'lr': 5e-3, 'maxiter': 100}
                )
        except (RuntimeError, NotPSDError) as e:  # Random acquisition
            if isinstance(e, RuntimeError) and not isinstance(e, NotPSDError):
                if e.args[0][:7] not in ['symeig_', 'cholesk']:
                    raise
            print_flush(f"\t\tNon PSD Error in GP fit. Re-fitting objective GP from scratch...")
            if args.self_training:
                print_flush('start retraining')
                unlabel_x, unlabel_y = train_teacher_student(args, X_train, y_train, X_train, y_train, bounds, datamodule)
                train_x_arg = torch.cat((train_x, unlabel_x),dim=0)
                y_train_arg = torch.cat((y_train, unlabel_y),dim=0)
                y_train_arg_std = y_train_arg.add(-y_train_arg.mean()).div(y_train_arg.std())
                gp_model = gp_torch_train(
                    train_x=train_x,
                    train_y=y_train_std,
                    # train_y=y_train_normalized,
                    n_inducing_points=current_n_inducing_points,
                    tkwargs=tkwargs,
                    init=True,
                    scale=args.scale,
                    covar_name=args.covar_name,
                    gp_file=gp_file,
                    save_file=new_gp_file,
                    input_wp=args.input_wp,
                    outcome_transform=None,
                    options={'lr': 5e-3, 'maxiter': 500}
                )
            else:
                gp_model = gp_torch_train(
                    train_x=train_x,
                    train_y=y_train_std,
                    # train_y=y_train_normalized,
                    n_inducing_points=current_n_inducing_points,
                    tkwargs=tkwargs,
                    init=True,
                    scale=args.scale,
                    covar_name=args.covar_name,
                    gp_file=gp_file,
                    save_file=new_gp_file,
                    input_wp=args.input_wp,
                    outcome_transform=None,
                    options={'lr': 5e-3, 'maxiter': 500}
                )
        curr_gp_file = new_gp_file

        # create bounds on posterior variance to use in acqf scheduling
        with torch.no_grad():
            y_pred_var = gp_model.posterior(train_x).variance
            yvarbounds = torch.zeros(2, y_train.shape[1], **tkwargs)
            yvarbounds[0] = torch.quantile(y_pred_var, .0005, dim=0)
            yvarbounds[1] = torch.quantile(y_pred_var, .9995, dim=0)
            yvardelta = .05 * (yvarbounds[1] - yvarbounds[0])
            yvarbounds[0] -= yvardelta
            yvarbounds[1] += yvardelta


        # Part 2: optimize GP acquisition func to query point
        # ===============================
        if pbar is not None:
            pbar.set_description("optimizing acq func")

        print_flush(f"\t\tPicking new inputs nb. {gp_iter + 1} via optimization...")
        try:  # BO acquisition
            print('robust_opt_topology', args.acq_func_id)
            res = bo_loop(
                gp_model=gp_model,
                acq_func_id=args.acq_func_id,
                acq_func_kwargs=args.acq_func_kwargs,
                acq_func_opt_kwargs=args.acq_func_opt_kwargs,
                bounds=normalize(bounds, bounds),
                tkwargs=tkwargs,
                q=q,
                num_restarts=args.num_restarts,
                raw_initial_samples=args.raw_initial_samples,
                seed=iter_seed,
                num_MC_sample_acq=args.num_MC_sample_acq,
            )
            z_opt = res
            rand_point_due_bo_fail += [0] * q
        except (RuntimeError, NotPSDError) as e:  # Random acquisition
            if isinstance(e, RuntimeError) and not isinstance(e, NotPSDError):
                if e.args[0][:7] not in ['symeig_', 'cholesk']:
                    raise
            print_flush(f"\t\tPicking new inputs nb. {gp_iter + 1} via random sampling...")
            n_rand_acq += q
            z_opt = torch.rand(q, bounds.shape[1]).to(bounds)
            exc = e
            rand_point_due_bo_fail += [1] * q

        z_opt = torch.atleast_2d(z_opt)
        # z_opt = unnormalize(z_opt, bounds).cpu().detach()

        model.to(**tkwargs)

                # Decode point
        smiles_opt, prop_opt = _batch_decode_z_and_props(
            model,
            unnormalize(z_opt, bounds),
            datamodule,
            invalid_score=invalid_score,
            pbar=pbar,
        )
        z_opt = unnormalize(z_opt, bounds).cpu().detach()
        # model.cpu()
        gp_model.cpu()
        gc.collect()
        torch.cuda.empty_cache()

        # Reset pbar description
        if pbar is not None:
            pbar.set_description(old_desc)

            # Update best point in progress bar
            if postfix is not None:
                postfix["best"] = max(postfix["best"], float(max(prop_opt)))
                pbar.set_postfix(postfix)

        # Append to new GP data
        latent_points = np.concatenate([latent_points, z_opt], axis=0)
        targets = np.concatenate([targets, prop_opt], axis=0)
        chosen_smiles.append(smiles_opt)
        _save_gp_data(latent_points, targets, chosen_smiles, gp_data_file)
        X_train = torch.cat([X_train, z_opt.to(**tkwargs)], dim=0)
        y_train = torch.cat([y_train, torch.tensor(prop_opt).view(-1,1).to(**tkwargs)], dim=0)
        # Append to overall list
        all_new_smiles += smiles_opt
        all_new_props += prop_opt
        print('--------------------------------------------')
        print(X_train.shape, y_train.shape, z_opt.shape)
        print(len(all_new_smiles), len(all_new_props))
        print('--------------------------------------------')

        if error_aware_acquisition:
            pass
        print('new score:', prop_opt, ' new smiles:', smiles_opt[0])
    # Update datamodule with ALL data points
    return all_new_smiles, all_new_props


def latent_sampling(args, model, datamodule, num_queries_to_do, pbar=None):
    """ Draws samples from latent space and appends to the dataset """

    z_sample = torch.randn(num_queries_to_do, model.latent_dim, device=model.device)
    z_decode, z_prop = _batch_decode_z_and_props(
        model, z_sample, datamodule, args, pbar=pbar
    )

    return z_decode, z_prop

def latent_turbo(model, 
                 args, 
                 tkwargs, 
                 datamodule: WeightedJTNNDataset, 
                 num_queries_to_do, 
                ):

    dset = datamodule.train_dataset
    q = 1
    model.to(**tkwargs)
    X_train, y_train, latent_points, targets, chosen_smiles = get_latent_dataset(model, 8000, 2000, dset, tkwargs)
    bounds, ybounds = get_bounds(X_train, y_train, tkwargs)
    X_train = X_train.detach().cpu().numpy()
    lb = X_train.min(0)
    ub = X_train.max(0)

    """ Draws samples from latent space and appends to the dataset """
    s = ScoreFunction(model, datamodule, args, tkwargs)
    turbo1 = Turbo1(
        f=s.score_function,  # Handle to objective function
        lb=lb,  # Numpy array specifying lower bounds
        ub=ub,  # Numpy array specifying upper bounds
        n_init=20,  # Number of initial bounds from an Latin hypercube design
        max_evals = num_queries_to_do,  # Maximum number of evaluations
        batch_size=10,  # How large batch size TuRBO uses
        verbose=True,  # Print information from each batch
        use_ard=True,  # Set to true if you want to use ARD for the GP kernel
        max_cholesky_size=2000,  # When we switch from Cholesky to Lanczos
        n_training_steps=50,  # Number of steps of ADAM to learn the hypers
        min_cuda=1024,  # Run on the CPU for small datasets
        device="cuda",  # "cpu" or "cuda"
        dtype="float32",  # float64 or float32
    )
    turbo1.optimize()

    return _batch_decode_z_and_props(
        model, torch.atleast_2d(torch.from_numpy(turbo1.X).to(**tkwargs)), datamodule, args.invalid_score
    )
    # model.decode_deterministic(z=torch.atleast_2d(torch.from_numpy(turbo1.X).to(**tkwargs))).detach().squeeze(0).cpu().numpy().reshape(-1, 40, 40), turbo1.fX.reshape(-1)

class ScoreFunction():
    def __init__(self, model, datamodule, args, tkwargs) -> None:
        super().__init__()
        self.args = args
        self.model = model
        self.tkwargs = tkwargs
        self.datamodule = datamodule
    def score_function(self, z):
        z = torch.atleast_2d(torch.from_numpy(z).to(**self.tkwargs))
        z_decode, z_prop = _batch_decode_z_and_props(
            self.model, z, self.datamodule, self.args.invalid_score)
        # print(z_decode, z_prop)
        return -z_prop[0]
        

if __name__ == "__main__":
    # Otherwise decoding fails completely
    rdkit_quiet()

    # Pytorch lightning raises some annoying unhelpful warnings
    # in this script (because it is set up weirdly)
    # therefore we suppress warnings
    # warnings.filterwarnings("ignore")

    main()
