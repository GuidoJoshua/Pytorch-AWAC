import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from src.utils.train_utils import soft_update


class BRAC(nn.Module):

    def __init__(self,
                 critic: nn.Module,  # Q(s, a)
                 critic_target: nn.Module,
                 actor: nn.Module,  # πθ(a|s)
                 dual_critic:nn.Module,
                 lam: float = 1.0,  # Regularization coefficient
                 alpha: float = 10.0,
                 gamma: float = 0.9,
                 num_action_samples: int = 1,
                 tau: float = 5e-3,
                 critic_lr: float = 3e-4,
                 actor_lr: float = 3e-4,
                 dual_lr: float = 3e-4,
                 use_adv: bool = False):
        super(BRAC, self).__init__()

        self.critic = critic
        self.critic_target = critic_target
        self.critic_target.load_state_dict(critic.state_dict())
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.actor = actor
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.dual_critic = dual_critic
        self.dual_critic_opt = torch.optim.Adam(self.dual_critic.parameters(), lr=dual_lr)
        
        self.lam = lam
        self.alpha = alpha
        self.gamma = gamma
        self.num_action_samples = num_action_samples
        self.tau = tau
        self.use_adv = use_adv

    def get_action(self, state, num_samples: int = 1):
        logits = self.actor(state)
        dist = Categorical(logits=logits)
        return dist.sample(sample_shape=[num_samples]).T



    def update_critic(self, state, action, reward, next_states, terms, truns):
        with torch.no_grad():
            qs = self.critic_target(next_states)  # [minibatch size x #.actions]
            sampled_as = self.get_action(next_states, self.num_action_samples)  # [ minibatch size x #. action samples]
            mean_qsa = qs.gather(1, sampled_as).mean(dim=-1, keepdims=True)  # [minibatch size x 1]
            q_target = reward + self.gamma * mean_qsa * (1 - terms - truns)

        q_val = self.critic(state).gather(1, action)
        loss = F.mse_loss(q_val, q_target)

        self.critic_opt.zero_grad()
        loss.backward()
        self.critic_opt.step()

        # target network update
        soft_update(self.critic, self.critic_target, self.tau)

        return loss



    def update_actor(self, state, action):

        # Sample actions from current policy
        logits = self.actor(state) # batch_size x num_actions
        log_prob = F.log_softmax(logits, dim=-1).gather(1, action) # batch size x 1
        qs = self.critic_target(state) # batch_size x num_actions
        qas = qs.gather(1, action) # batch_size x 1
        
        
        loss = -1 * torch.mean(log_prob * qas)
        
        action_p = self.get_action(state)  # Actions from policy
        action_b = action  # Actions from behavior policy

        # Compute Wasserstein Distance
        logits_p = self.dual_critic(state, action_p)
        logits_b = self.dual_critic(state, action_b)
        W = torch.mean(logits_p) - torch.mean(logits_b)

        # Compute Q-values for actions sampled from the policy
        # Actor Loss
        actor_loss = loss + self.alpha * W

        # Optimize actor
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        return actor_loss

    
    def update_dual_critic(self, state, action):
        with torch.no_grad():
            action_p = self.get_action(state)
        action_b = action
        alpha = torch.rand(action_b.size(0), 1, device=state.device)
        interpolated_actions = alpha * action_p + (1 - alpha) * action_b
        interpolated_actions.requires_grad_(True)
        interpolated_critic = self.dual_critic(state, interpolated_actions)
        gradients = torch.autograd.grad(
            outputs=interpolated_critic,
            inputs=interpolated_actions,
            grad_outputs=torch.ones_like(interpolated_critic),
            create_graph=True,
            retain_graph=True,
        )[0]
        gradient_penalty = self.alpha * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        dual_critic_loss = -1 * torch.mean(self.dual_critic(state, action_p) - self.dual_critic(state, action_b) + gradient_penalty)
        self.dual_critic_opt.zero_grad()
        dual_critic_loss.backward()
        self.dual_critic_opt.step()