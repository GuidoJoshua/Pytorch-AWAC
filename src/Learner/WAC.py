import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from src.utils.train_utils import soft_update


class WAC(nn.Module):

    def __init__(self,
                 critic: nn.Module,
                 critic_target: nn.Module,
                 actor: nn.Module,
                 dual_critic:nn.Module,
                 lam: float = 1.0,
                 alpha: float = 10.0,
                 gamma: float = 0.9,
                 num_action_samples: int = 1,
                 tau: float = 5e-3,
                 critic_lr: float = 1e-4,
                 actor_lr: float = 3e-4,
                 dual_lr: float = 3e-4,
                 use_adv: bool = False):
        super(WAC, self).__init__()

        self.critic = critic
        self.critic_target = critic_target
        self.critic_target.load_state_dict(critic.state_dict())
        self.critic_opt = torch.optim.RMSprop(self.critic.parameters(), lr=critic_lr)

        self.actor = actor
        self.actor_opt = torch.optim.RMSprop(self.actor.parameters(), lr=actor_lr)

        self.dual_critic = dual_critic
        self.dual_critic_opt = torch.optim.RMSprop(self.dual_critic.parameters(), lr=dual_lr)
        
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



    def update_actor(self, state, action, n_step, total_step, verbose=False):
        logits = self.actor(state) 
        log_prob = F.log_softmax(logits, dim=-1).gather(1, action)
        prob = F.softmax(logits, dim=-1)
        qs = self.critic_target(state)
        vs = (qs * prob).sum(dim=-1, keepdims=True)
        qas = qs.gather(1, action)
        loss = -1.0 * torch.mean(log_prob * qas) 
    
        # Wasserstein distance
        action_p = self.get_action(state)  # policy
        action_b = action                 # behavior
        logits_p = self.dual_critic(state, action_p).mean()
        logits_b = self.dual_critic(state, action_b).mean()
        W_distance = logits_b - logits_p
        
        actor_loss = loss + self.alpha * W_distance

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
        if verbose:
            print(f"sampled action: {action[:5]}")
            print(f"p(a|s): {prob[:5]} \n Q: {qs[:5]}")
            print(f"loss: {loss} \n W: {W_distance}")
        return actor_loss


    
    def update_dual_critic(self, state, action):
        with torch.no_grad():
            action_p = self.get_action(state)
            action_b = action
        D_behavior = self.dual_critic(state, action_b).mean()
        D_policy = self.dual_critic(state, action_p).mean()
        w_distance = D_behavior - D_policy 


        base_loss = - w_distance

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

        gp_weight = 10.0
        
        # Lip-1 condition
        gradient_penalty = gp_weight * ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        dual_critic_loss = base_loss + gradient_penalty

        self.dual_critic_opt.zero_grad()
        dual_critic_loss.backward()
        self.dual_critic_opt.step()

        return dual_critic_loss
