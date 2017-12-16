import gym
from skimage.transform import resize
from skimage.color import rgb2grey
import argparse
import os
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

from networks import EnvModel, I3A
from storage import RolloutStorage
import utils
from envs import make_env

# Utility functions from baselines github repo
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv


def process_state(frame):
    """Processes a frame, converts it into grey scale and
       resizes to a 50 by 50 image"""
    frame_crop = frame[10:185, 5:, :]
    frame_grey = rgb2grey(resize(frame_crop, (50, 50)))
    return frame_grey

def logprobs_and_entropy(logits, actions):
    log_probs = F.log_softmax(logits)
    probs = F.softmax(logits)

    dist_entropy = -(probs * log_probs).sum(-1).mean()
    action_log_prob = log_probs.gather(1, actions)
    return action_log_prob, dist_entropy


if __name__ == '__main__':
    # set up argument parser
    parser = argparse.ArgumentParser()

    # experiment
    parser.add_argument('--exp', default = 'test')
    parser.add_argument('--resume', default = None, type = str)
    parser.add_argument('--clean', action = 'store_true')
    parser.add_argument('--seed', default = 1, type = int)

    # dataset
    parser.add_argument('--env', default = 'Frostbite-v0')
    # parser.add_argument('--env', default = 'SpaceInvaders-v0')

    # training
    parser.add_argument('--num-frames', default = 1e7, type = int)
    parser.add_argument('--minibatch', default = 4, type = int)
    parser.add_argument('--eval', default = 500, type = int)
    parser.add_argument('--snapshot', default = 500, type = int)
    parser.add_argument('--gpu', default = '7')
    parser.add_argument('--env-path', default = None)
    parser.add_argument('--num-train', default = 8, type = int,
                        help="Number of different gym instances to gen data")
    parser.add_argument('--forward-steps', default = 128, type = int,
                        help="Number of forward frames to simulate")

    # Reinforcement Learning Parameters
    parser.add_argument('--ppo-clip_param', default = 0.1,
                        help="Clip parameter for PPO")
    parser.add_argument('--ppo-epoch', default = 3, type = int,
                        help="Number of times to process data")
    parser.add_argument('--tau', type=float, default=0.95,
                        help='gae parameter (default: 0.95)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='decay rate')
    parser.add_argument('--entropy-coef', type=float, default=0.01,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--value-loss-coef', type=float, default=1.0,
                        help='value loss coefficient (default: 0.5)')

    # Optimization Parameters
    parser.add_argument('--lr', default = 2.5e-4, type = float)
    parser.add_argument('--momentum', default = 0.9, type = float)
    parser.add_argument('--weight_decay', default = 1e-5, type = float)
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='max norm of gradients (default: 0.5)')

    # parse arguments
    args = parser.parse_args()
    print('==> arguments parsed')

    # set up gpus for training
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # set up experiment path
    exp_path = os.path.join('exp', args.exp)
    utils.shell.mkdir(exp_path, clean = args.clean)
    logger = utils.Logger(exp_path)
    print('==> save logs to {0}'.format(exp_path))

    # log parameters
    with open(os.path.join(exp_path, "exp.info"), "w") as f:
        for key in vars(args):
            print('[{0}] = {1}'.format(key, getattr(args, key)))
            print('[{0}] = {1}'.format(key, getattr(args, key)), file=f)
    args.num_frames = int(args.num_frames)

    # Set up Environment
    envs = [make_env(args.env, args.seed, i) for i in range(args.num_train)]
    envs = SubprocVecEnv(envs)
    # Also set up Environment for Evaluation with Unclipped Rewards
    eval_env = make_env(args.env, args.seed, args.num_train, eval=True)()
    print('==> environment setup')

    obs_shape = envs.observation_space.shape

    env_model = EnvModel()
    if args.env_path:
        model_params = torch.load(args.env_path)
        env_model.load_state_dict(model_params)

    # Don't train the environmental model
    env_model.eval()
    env_model.cuda()
    model = I3A(env_model=env_model, actions=envs.action_space.n)
    model.cuda()

    # set up optimizer for training
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    print('==> optimizer loaded')

    # load snapshot of model and optimizer
    if args.resume is not None:
        if os.path.isfile(args.resume):
            snapshot = torch.load(args.resume)
            epoch = snapshot['epoch']
            model.load_state_dict(snapshot['model'])
            # If this doesn't work, can use optimizer.load_state_dict
            optimizer.load_state_dict(snapshot['optimizer'])
            print('==> snapshot "{0}" loaded (epoch {1})'.format(args.resume, epoch))
        else:
            raise FileNotFoundError('no snapshot found at "{0}"'.format(args.resume))
    else:
        epoch = 0

    # State size is represented is 200 hidden size, 200 cell size
    rollouts = RolloutStorage(args.forward_steps, args.num_train, obs_shape, envs.action_space, 2 * model.state_size)
    current_obs = torch.zeros(args.num_train, *obs_shape)
    num_updates = args.num_frames // args.num_train // args.forward_steps
    frame_num = 0

    def update_current_obs(obs):
        shape_dim0 = envs.observation_space.shape[0]
        obs = torch.from_numpy(obs).float()
        current_obs[:, -shape_dim0:] = obs

    obs = envs.reset()
    update_current_obs(obs)

    rollouts.observations[0].copy_(current_obs)

    # These variables are used to compute average rewards for all processes.
    episode_rewards = torch.zeros([args.num_train, 1])
    final_rewards = torch.zeros([args.num_train, 1])

    current_obs = current_obs.cuda()
    rollouts.cuda()

    for j in range(num_updates):
        model.train()
        for step in range(args.forward_steps):
            # Sample actions
            hx, cx = rollouts.states[step].split(model.state_size, 1)
            value, action_logit, o_logit, states = model.forward((Variable(rollouts.observations[step], volatile=True),
                                                                      (Variable(hx, volatile=True), Variable(cx, volatile=True)),
                                                                      Variable(rollouts.masks[step], volatile=True)))
            action_probs = F.softmax(action_logit)
            log_probs = F.log_softmax(action_logit)
            actions = action_probs.multinomial()
            action_log_probs = log_probs.gather(1, actions)
            cpu_actions = actions.data.squeeze(1).cpu().numpy()

            # Obser reward and next obs
            obs, reward, done, info = envs.step(cpu_actions)
            reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()
            episode_rewards += reward

            print(done)

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            final_rewards *= masks
            final_rewards += (1 - masks) * episode_rewards
            episode_rewards *= masks

            masks = masks.cuda()

            if current_obs.dim() == 4:
                current_obs *= masks.unsqueeze(2).unsqueeze(2)
            else:
                current_obs *= masks

            states = torch.cat([state.data for state in states], 2)

            update_current_obs(obs)
            rollouts.insert(step, current_obs, states.squeeze(0), actions.data, action_log_probs.data, value.data, reward, masks)

        hx, cx = rollouts.states[-1].split(model.state_size, 1)
        next_value = model.forward((Variable(rollouts.observations[-1], volatile=True),
                                  (Variable(hx), Variable(cx)),
                                  Variable(rollouts.masks[-1], volatile=True)))[0].data

        rollouts.compute_returns(next_value, args.gamma, args.tau)

        # After constructing points, now train points
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        for e in range(args.ppo_epoch):
            data_generator = rollouts.recurrent_generator(advantages,
                                                    args.minibatch)

            for sample in data_generator:
                observations_batch, states_batch, actions_batch, \
                   return_batch, masks_batch, old_action_log_probs_batch, \
                        adv_targ = sample

                # Reshape to do in a single forward pass for all steps
                hx, cx = states_batch.split(model.state_size, 1)
                values, logits, _, states = model.forward((Variable(observations_batch),
                                                                                               (Variable(hx), Variable(cx)),
                                                                                               Variable(masks_batch)))

                # Compute action probabilities and entropy distance
                action_log_probs, dist_entropy = logprobs_and_entropy(logits, Variable(actions_batch))

                adv_targ = Variable(adv_targ)
                ratio = torch.exp(action_log_probs - Variable(old_action_log_probs_batch))
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - args.ppo_clip_param, 1.0 + args.ppo_clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean() # PPO's pessimistic surrogate (L^CLIP)

                value_loss = (Variable(return_batch) - values).pow(2).mean()

                optimizer.zero_grad()
                loss = (value_loss + action_loss - dist_entropy * args.entropy_coef)
                loss.backward()
                nn.utils.clip_grad_norm(model.parameters(), args.max_grad_norm)
                optimizer.step()

        rollouts.after_update()
        frame_num += args.forward_steps * args.num_train
        logger.scalar_summary('Loss', loss.data[0], frame_num)
        logger.scalar_summary('Value Loss', value_loss.data[0], frame_num)
        logger.scalar_summary('Action Loss', action_loss.data[0], frame_num)
        logger.scalar_summary('Entropy Value', dist_entropy.data[0], frame_num)
        logger.scalar_summary('Mean Rewards', final_rewards.mean(), frame_num)
        logger.scalar_summary('Max Rewards', final_rewards.max(), frame_num)

        if j % args.eval == 0:
            state = eval_env.reset()
            model.eval()
            score = 0
            hx, cx = Variable(torch.zeros(1, 1, model.state_size)).cuda(), Variable(torch.zeros(1, 1, model.state_size)).cuda()
            mask = Variable(torch.zeros(1, 1, 1)).cuda()
            while True:
                state = Variable(torch.from_numpy(state).unsqueeze(0).float()).cuda()

                # Action logits
                action_logit = model.forward((state, (hx, cx), mask))[1]
                action_logit = action_logit.data.numpy()
                action = action_logit.argmax(axis=1)[0, 0]

                state, reward, done, _ = eval_env.step(action)
                score += reward
                if done:
                    break
            logger.scalar_summary('Actual Score', score, frame_num)
            print(' * Actual Score of {}'.format(score))

        if j % args.snapshot == 0:
            snapshot = {
                'epoch': j + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            torch.save(snapshot, os.path.join(exp_path, 'epoch-{}.pth'.format(j)))
            print('==> saved snapshot to "{0}"'.format(os.path.join(exp_path, 'model_{}.pth'.format(frame_num))))
