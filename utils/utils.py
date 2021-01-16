import torch
import copy
import numpy as np

cuda_available = torch.cuda.is_available()


def xent_loss(out_1, out_2, args):
    # [2*B, D]
    out = torch.cat([out_1, out_2], dim=0)
    # [2*B, 2*B]
    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / args.temperature)
    mask = (
        torch.ones_like(sim_matrix)
        - torch.eye(2 * args.batch_size, device=sim_matrix.device)
    ).bool()
    # [2*B, 2*B-1]
    sim_matrix = sim_matrix.masked_select(mask).view(2 * args.batch_size, -1)

    # compute loss
    pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / args.temperature)
    # [2*B]
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    return - torch.log(pos_sim / sim_matrix.sum(dim=-1))


def compute_jacobian(inputs, output):
    """
    :param inputs: Batch X Size (e.g. Depth X Width X Height)
    :param output: Batch X Classes
    :return: jacobian: Batch X Classes X Size
    """
    assert inputs.requires_grad

    num_classes = output.size()[1]

    jacobian = torch.zeros(num_classes, *inputs.size())
    grad_output = torch.zeros(*output.size())
    if inputs.is_cuda:
        grad_output = grad_output.cuda()
        jacobian = jacobian.cuda()

    for i in range(num_classes):
        zero_gradients(inputs)
        grad_output.zero_()
        grad_output[:, i] = 1
        output.backward(grad_output, retain_graph=True)
        jacobian[i] = inputs.grad.data

    return torch.transpose(jacobian, dim0=0, dim1=1)


# get color jitter tensors
def get_batch_color_jitter_tensors(net, shape, eps=1e-3):
    jit_params = net.augment.jit.generate_parameters(shape)
    jit_params_delta = net.augment.jit.generate_parameters(shape)
    for k in jit_params.keys():
        if k is "order":
            jit_params_delta[k] = jit_params[k]
        else:
            jit_params_delta[k] = jit_params[k] + eps

    if cuda_available:
        for k in jit_params.keys():
            jit_params[k] = jit_params[k].cuda()
            jit_params_delta[k] = jit_params_delta[k].cuda()

    return jit_params, jit_params_delta


def get_batch_op_augment_params(
    op, shape, eps, only_keys=None, clamp_low=0, clamp_hi=31
):
    # import pdb; pdb.set_trace()
    params = op.generate_parameters(shape)
    params_delta = op.generate_parameters(shape)
    if type(params) == dict:
        for k in params.keys():
            if only_keys is not None and k in only_keys:
                params_delta[k] = params[k] + eps
                #  Needs to be generalized, currently using this for only crop
                if type(params_delta[k]) == np.ndarray:
                    params_delta[k] = np.clip(params_delta[k], clamp_low, clamp_hi)
                else:
                    params_delta[k] = params_delta[k].clamp(clamp_low, clamp_hi)
            else:
                if type(params[k]) == list or type(params[k]) == np.ndarray:
                    params_delta[k] = params[k]
                elif (
                    params[k].dtype == torch.float64 or params[k].dtype == torch.float32
                ):
                    params_delta[k] = params[k] + eps
                else:
                    params_delta[k] = params[k]

        if cuda_available:
            for k in params.keys():
                if type(params[k]) == np.ndarray:
                    continue
                params[k] = params[k].cuda()
                params_delta[k] = params_delta[k].cuda()

    return params, params_delta


def get_batch_op_augment_params_centered(
    op, shape, eps, only_keys=None, clamp_low=0, clamp_hi=31
):
    params = op.generate_parameters(shape)
    params_delta_r = op.generate_parameters(shape)
    params_delta_l = op.generate_parameters(shape)
    if type(params) == dict:
        for k in params.keys():
            if only_keys is not None and k in only_keys:
                params_delta_r[k] = params[k] + eps
                params_delta_l[k] = params[k] - eps
                params_delta_r[k] = params_delta_r[k].clamp(clamp_low, clamp_hi)
                params_delta_l[k] = params_delta_l[k].clamp(clamp_low, clamp_hi)
            else:
                if type(params[k]) == np.ndarray:
                    params_delta_r[k] = params[k]
                    params_delta_l[k] = params[k]
                elif (
                    params[k].dtype == torch.float64 or params[k].dtype == torch.float32
                ):
                    params_delta_r[k] = params[k] + eps
                    params_delta_l[k] = params[k] - eps
                else:
                    params_delta_r[k] = params[k]
                    params_delta_l[k] = params[k]

        if cuda_available:
            for k in params.keys():
                params[k] = params[k].cuda()
                params_delta_r[k] = params_delta_r[k].cuda()
                params_delta_l[k] = params_delta_l[k].cuda()

    return params, params_delta_r, params_delta_l


def get_batch_augmentation_centered_params(net, shape, eps=1e-3):
    params = {}
    params_delta_r, params_delta_l = {}, {}
    # Generate params for cropping
    (
        params["crop_params"],
        params_delta_r["crop_params_delta"],
        params_delta_l["crop_params_delta"],
    ) = get_batch_op_augment_params_centered(
        net.augment.crop, shape, eps=1, only_keys=["top_x", "top_y"]
    )

    # Generate params for horizontal flip
    (
        params["hor_flip_params"],
        params_delta_r["hor_flip_params_delta"],
        params_delta_l["hor_flip_params_delta"],
    ) = get_batch_op_augment_params_centered(net.augment.hor_flip, shape, eps)
    hor_flip_probs = torch.rand(shape[0])
    params["hor_flip_params"]["batch_prob"] = hor_flip_probs < net.augment.hor_flip_prob

    # Generate params for color jitter
    jit_probs = torch.rand(shape[0])
    B = (jit_probs < net.augment.jit_prob).sum()
    jit_probs_shape = (B, 3, 32, 32)
    (
        params["jit_params"],
        params_delta_r["jit_params_delta"],
        params_delta_l["jit_params_delta"],
    ) = get_batch_op_augment_params_centered(net.augment.jit, jit_probs_shape, eps)
    params["jit_batch_probs"] = jit_probs < net.augment.jit_prob

    # Generate params for random grayscaling
    (
        params["grayscale_params"],
        params_delta_r["grayscale_params_delta"],
        params_delta_l["grayscale_params_delta"],
    ) = get_batch_op_augment_params_centered(net.augment.rand_grayscale, shape, eps)
    gs_probs = torch.rand(shape[0])
    params["grayscale_params"]["batch_prob"] = gs_probs < net.augment.gs_prob

    return params, params_delta_r, params_delta_l


# get color jitter tensors
def get_batch_augmentation_params(net, shape, eps=1e-3):
    params = {}
    params_delta = {}
    # Generate params for cropping
    (
        params["crop_params"],
        params_delta["crop_params_delta"],
    ) = get_batch_op_augment_params(
        net.augment.crop, shape, eps=1, only_keys=["top_x", "top_y"]
    )

    # Generate params for horizontal flip
    (
        params["hor_flip_params"],
        params_delta["hor_flip_params_delta"],
    ) = get_batch_op_augment_params(net.augment.hor_flip, shape, eps)
    hor_flip_probs = torch.rand(shape[0])
    params["hor_flip_params"]["batch_prob"] = hor_flip_probs < net.augment.hor_flip_prob

    # Generate params for color jitter
    # parameters for color jitter
    jit_probs = torch.rand(shape[0])
    B = (jit_probs < net.augment.jit_prob).sum()
    jit_probs_shape = (B, 3, 32, 32)
    (
        params["jit_params"],
        params_delta["jit_params_delta"],
    ) = get_batch_op_augment_params(net.augment.jit, jit_probs_shape, eps)
    params["jit_batch_probs"] = jit_probs < net.augment.jit_prob

    # Generate params for random grayscaling
    (
        params["grayscale_params"],
        params_delta["grayscale_params_delta"],
    ) = get_batch_op_augment_params(net.augment.rand_grayscale, shape, eps)
    gs_probs = torch.rand(shape[0])
    params["grayscale_params"]["batch_prob"] = gs_probs < net.augment.gs_prob
    return params, params_delta


def get_jitter_norm_loss(net, pos, out, params, params_delta, eps):
    # compute approximate derivative wrt jitter
    jitter_params = copy.deepcopy(params)
    jitter_params["jit_params"] = params_delta["jit_params_delta"]
    _, out_djitter = net(pos, jitter_params)
    j_djitter = torch.mean(torch.norm((out_djitter - out) / eps, dim=1))
    return j_djitter


def get_jitter_norm_loss_centered(
    net, pos, params, params_delta_r, params_delta_l, eps
):
    # compute approximate derivative wrt jitter
    jitter_params = copy.deepcopy(params)
    jitter_params["jit_params"] = params_delta_r["jit_params_delta"]
    _, out_djitter_r = net(pos, jitter_params)
    jitter_params["jit_params"] = params_delta_l["jit_params_delta"]
    _, out_djitter_l = net(pos, jitter_params)
    j_djitter = torch.mean(
        torch.norm((out_djitter_r - out_djitter_l) / (2 * eps), dim=1)
    )
    return j_djitter


def get_crop_norm_loss(net, pos, out, params, params_delta, eps):
    # compute approximate derivative wrt crop params
    crop_params = copy.deepcopy(params)
    crop_params["crop_params"] = params_delta["crop_params_delta"]
    _, out_dcrop = net(pos, crop_params)
    j_dcrop = torch.mean(torch.norm((out_dcrop - out) / eps, dim=1))
    return j_dcrop


def get_crop_norm_loss_centered(net, pos, params, params_delta_r, params_delta_l, eps):
    # compute approximate derivative wrt crop params
    crop_params = copy.deepcopy(params)
    crop_params["crop_params"] = params_delta_r["crop_params_delta"]
    _, out_dcrop_r = net(pos, crop_params)
    crop_params["crop_params"] = params_delta_l["crop_params_delta"]
    _, out_dcrop_l = net(pos, crop_params)
    j_dcrop = torch.mean(torch.norm((out_dcrop_r - out_dcrop_l) / (2 * eps), dim=1))
    return j_dcrop


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
