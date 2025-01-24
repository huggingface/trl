# PRIME Trainer

The Process Reinforcement through IMplicit rEwards (PRIME) algorithm by Cui et al. starts from an SFT policy, Process Reward Model (PRM), and a frozen reference model together with a ground truth outcome verifier. For each RL iteration, the policy model first generates rollouts. Then, the implicit PRM and outcome verifier score the rollouts, and the implicit PRM get updated on the rollouts with outcome reward. Finally, the outcome reward and process reward are combined and used to update the policy model via a PPO loss update. More information can be found in the [PRIME Blog](https://curvy-check-498.notion.site/Process-Reinforcement-through-Implicit-Rewards-15f4fcb9c42180f1b498cc9b2eaf896f).

## Algorithm

The algorithm for PRIME is descibed as follows:

**Input**: Input supervised fine-tuned model $\pi_{\theta_{\mathrm{init}}}$; ground truth outcome reward verifier function $R_\mathrm{Gt}$; implicit PRM $\pi_\phi$; frozen reference model $\pi_{\mathrm{ref}}$; instruction dataset with ground truth outputs $\mathcal{D} = \{(x, y_\textrm{gt})\}$; and hyperparameters $\beta, \epsilon, M, N, K, \gamma$.

**Notation**: $r_i$ represents the outcome reward of $y_i$ the $i$-th candidate output, and $r_i^t$ denotes its process reward at token step $t$. Define
$r_\phi (y) = \beta \log \frac{\pi_\phi (y)}{\pi_{\theta_{\mathrm{ref}}} (y)}$, where the context $x$ is omitted for simplicity.

**Steps**:

1. Initialize policy model $\phi_\theta \leftarrow \pi_{\theta_{\mathrm{init}}}$, implicit PRM $\pi_{\phi} \leftarrow	 \pi_{\theta_{\mathrm{init}}}$, and  reference model $\pi_{\mathrm{ref}} \leftarrow	  \pi_{\theta_{\mathrm{init}}}$ from the initial SFT model $\pi_{\theta_{\mathrm{init}}}$

2. For iterations $1, \ldots, N$ do:
    1. Sample batch of prompts $\mathcal{B} \sim \mathcal{D}$
    2. Initialize the buffer $\mathcal{T} = \emptyset$
    3. for each prompt instruction $(x, y_\textrm{gt}) \in \mathcal{B}$ do:
        1. Sample $K$ candidate outputs from the current policy $\pi_\theta$:  $y_1, \ldots, y_K \sim \pi_{\theta}(\cdot | x)$
        2. Compute ground truth rewards $r_i = R_\mathrm{Gt}(x, y_i, y_\textrm{gt})$ for $i = 1, \ldots, K$
        3. Record the number of correct responses $|\mathcal{C}_x|  = |\{y_i | r_i =1\}|$
        4. if the number of correct response ratio $0.2 <|\mathcal{C}_x| / K < 0.8$ is between 0.2 and 0.8: 
            1. add ALL the $K$ tuples of prompt $x$ and candidate outputs and ground truth rewards: $\{(x, y_i, r_i)\}_{i=1}^K$ to $\mathcal{T}$ 
        5. else:
            1. drop this prompt instruction $x$ and continue to the next prompt
    4. For PPO epoch $1, \ldots, M$ do:
        1. Forward pass the implicit PRM $\pi_\phi$ on each $(x, y, r) \in \mathcal{T}$ to get the implicit process rewards as $r^t = \beta \log \frac{\pi_\phi(y_t | y_{<t})}{\pi_{\mathrm{ref}}(y_t | y_{<t})}$ for each token $t$ of $y$
        2. Update the Implicit PRM $\pi_\phi$ using the Cross-Entropy loss given the tuples $(x, y, r)$:
        $\mathcal{L}_{\mathrm{CE}}(\phi) = \mathbb{E}_{(x, y, r) \sim \mathcal{T}} \left[ r \log \sigma (r_\phi (y)) + (1-r) \log (1-\sigma (r_\phi (y))) \right]$
        3. Compute the RLOO Advantage $A$ for the prompt $x$ and its $K$ candidate outputs $\{y_1, ..., y_i, ..., y_K\}$ and ground outcome rewards $\{ r_1, ..., r_i, ..., r_K \}$: for each token $t$ of $y_i$ let $A_{i}^t$ be advantage of the sum of the RLOO wth outcome rewards and the RLOO with implicit process rewards: 
        $ A_{i}^t  = r_i - \frac{1}{K -1} \sum_{j \neq i} r_j  + \sum_{s=t}^{|y_i|} \gamma^{s-t} \left[ 
            r_i^s - \frac{1}{K -1} \sum_{j \neq i} \frac{r_\phi(y_j)}{|y_j|}
        \right] $
        4. update the policy $\pi_\theta$ using the PPO loss with respect to the RLOO Advantage $A$ and $\pi_{\mathrm{ref}}$ with clip coefficient $\epsilon$. 

**Output**: return the final policy $\pi_\theta$ for saving.

# Get started

To just run a PRIME script to make sure the trainer can run, you can run the following command to train a PRIME model:

```bash
```

## PrimeTrainer

[[autodoc]] PrimeTrainer

## PrimeConfig

[[autodoc]] PrimeConfig
