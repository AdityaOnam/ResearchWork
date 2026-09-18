# Codebase Architecture & File Connectivity

Here is a visual map of how all the files in your repository connect to form the complete training and inference pipeline:

```mermaid
graph TD
    %% Execution Layer
    subgraph ExecutionLayer
        train_file["train_rl_model.py"]
        eval_file["validate.py and evaluate.py"]
        config_file["config/default_config.yaml"]
    end

    %% Data Layer
    subgraph DataLayer
        dataset_file["data/dataset.py"]
        curriculum_file["data/noise_curriculum.py"]
        noise_file["data/noise_injection.py"]
    end

    %% Model Layer
    subgraph ModelLayer
        iter_model_file["models/iterative_model.py"]
        ann_file["models/ann_prior.py"]
        vit_file["models/vit_backbone.py"]
        pde_file["models/pde_block.py"]
        denoiser_file["models/denoiser.py"]
        decoder_file["models/decoder.py"]
    end

    %% RL Layer
    subgraph RLLayer
        agent_file["rl/rl_agent.py"]
        policy_file["rl/policy.py"]
        reward_file["rl/reward.py"]
        buffer_file["rl/replay_buffer.py"]
    end

    %% Optimization Layer
    subgraph OptimizationLayer
        loss_file["losses/losses.py"]
        utils_file["utils/checkpointing.py"]
    end

    %% Connections
    config_file --> train_file
    config_file --> eval_file
    
    train_file --> dataset_file
    dataset_file --> curriculum_file
    curriculum_file --> noise_file
    noise_file -.->|Noisy Data| iter_model_file
    
    train_file --> iter_model_file
    train_file --> agent_file
    train_file --> loss_file
    train_file --> utils_file

    iter_model_file --> ann_file
    iter_model_file --> vit_file
    iter_model_file --> pde_file
    iter_model_file --> denoiser_file
    iter_model_file --> decoder_file
    
    agent_file --> policy_file
    agent_file --> buffer_file
    iter_model_file -.->|Predictions| reward_file
    reward_file -.->|Reward Signal| agent_file
    policy_file -.->|Actions| denoiser_file
```

## Detailed File-by-File Breakdown & Usage

### 1. Execution Scripts (Entry Points)
These are the files you run directly from your terminal using `python <filename>`.

*   **`train_rl_model.py`**
    *   **What happens:** This is the master loop. It initializes the PyTorch neural network (`IterativeRLModel`), the PPO Reinforcement Learning Agent, and the dataloaders. It loops over epochs, runs forward passes, computes rewards, updates the RL policy, and saves checkpoints.
    *   **How to use it:** `python train_rl_model.py` (Run this when you want to train the full physics-informed RL pipeline).
*   **`train_warmup.py`**
    *   **What happens:** Runs the same model but *disables* the RL agent. It trains the neural network using standard supervised losses (MAE) to give the network a solid foundational understanding of brain anatomy before the RL agent is allowed to start tweaking things.
    *   **How to use it:** `python train_warmup.py` (Run this first on a new dataset).
*   **`validate.py` & `evaluate.py`**
    *   **What happens:** Loads a saved `.pth` checkpoint and runs it on the unseen test dataset. It calculates metrics (like FW MAE, ICVF consistency) and does not calculate gradients or update weights.
    *   **How to use it:** `python validate.py` (Run this to see how well a trained model performs).
*   **`config/default_config.yaml`**
    *   **What happens:** Holds every hyperparameter (learning rates, batch size, noise levels, network depth). 
    *   **How to use it:** Edit this file directly in VS Code to tweak parameters before starting a training run.

### 2. Data Layer (`data/`)
These files run implicitly when `train_rl_model.py` requests data.

*   **`data/dataset.py`**
    *   **What happens:** Subclasses `torch.utils.data.Dataset`. It reads the raw `.nii.gz` or `.h5` files from the hard drive, normalizes the signals, and yields dictionaries containing `dwi`, `fw_gt`, `icvf_gt`, and `tissue_masks`.
*   **`data/noise_curriculum.py`**
    *   **What happens:** Takes the current epoch number and decides the target SNR. For example, it might dictate SNR=0 for epochs 1-10 (extreme noise), and gradually increase it to SNR=100 by epoch 50.
*   **`data/noise_injection.py`**
    *   **What happens:** Contains the mathematical Rician noise physics. It takes the clean DWI slice and the SNR from the curriculum, calculates the exact standard deviation ($\sigma$) required, and corrupts the signal.
*   **`data/synthetic_generator.py`**
    *   **What happens:** A standalone utility that creates fake geometric "phantoms" (circles, squares) to test if the model's PDE logic works in a controlled environment without real brain complexity.
    *   **How to use it:** `python data/synthetic_generator.py` (Run to generate test datasets).

### 3. Model Layer (`models/`)
These files execute during the `model.forward()` pass.

*   **`models/iterative_model.py`**
    *   **What happens:** The conductor of the forward pass. It loops $K$ times (e.g., 8 iterations). It feeds data into the ViT, passes that to the PDE block, gets an intermediate prediction, applies the denoiser, and repeats.
*   **`models/ann_prior.py`**
    *   **What happens:** A simple Multi-Layer Perceptron (MLP) that looks at a single voxel's 91 diffusion directions and makes a rough guess of what the FW and ICVF should be, completely ignoring surrounding neighboring pixels.
*   **`models/vit_backbone.py`**
    *   **What happens:** A Vision Transformer (`SimpleViT`). It cuts the image into patches and uses self-attention to understand the global structure of the brain slice (e.g., "this is the corpus callosum").
*   **`models/pde_block.py`**
    *   **What happens:** Implements differential equations. It takes the features and diffuses (smoothes) them. Crucially, it reads the `tissue_masks` so it *stops* smoothing when it hits the boundary between White Matter and CSF, preserving sharp anatomical edges.
*   **`models/denoiser.py`**
    *   **What happens:** Takes the raw, noisy DWI signal and cleans it slightly between iterations. The strength of this cleaning is dictated entirely by the RL agent.
*   **`models/decoder.py`**
    *   **What happens:** Takes all the processed features from all 8 iterations and projects them down into the final 2D images (the FW and ICVF maps).

#### The Iterative Feedback Loop (Visualized)
This diagram shows how the model feeds its own predictions back into itself across the 8 loops inside `iterative_model.py`.

```mermaid
graph TD
    %% Initial Inputs
    DWI_Raw["Raw Noisy DWI (91 ch)"]
    FW_Prior["ANN FW Prior (1 ch)"]
    ICVF_Prior["ANN ICVF Prior (1 ch)"]

    %% Iteration Start
    Concat(("Concatenate (93 ch)"))
    
    DWI_Raw -->|Iteration 0 Only| Concat
    FW_Prior -->|Iteration 0 Only| Concat
    ICVF_Prior -->|Iteration 0 Only| Concat

    subgraph "Iterative Loop (Repeats 8 Times)"
        Concat --> ViT["ViT Backbone (Global Context)"]
        ViT --> PDE["Tissue-Adaptive PDE (Physics Smoothing)"]
        
        %% Intermediate Decoding
        PDE --> IntDecoder["Intermediate Decoder"]
        IntDecoder --> FW_k["Updated FW Prediction"]
        IntDecoder --> ICVF_k["Updated ICVF Prediction"]
        
        %% RL Denoising
        PDE --> RL_Denoiser["RL-Controlled Denoiser"]
        RL_Denoiser --> DWI_k["Slightly Denoised DWI"]
    end

    %% Feedback Loop mechanism
    FW_k -.->|"Feed to Iteration K+1"| Concat
    ICVF_k -.->|"Feed to Iteration K+1"| Concat
    DWI_k -.->|"Feed to Iteration K+1"| Concat

    %% Final Exit
    PDE ===>|"After 8 Loops"| FinalDecoder["Multi-Scale Final Decoder"]
    FinalDecoder --> FinalOut["Final Crisp FW & ICVF Maps"]
```

### 4. Reinforcement Learning Layer (`rl/`)
Executes alongside the model to optimize decision-making.

*   **`rl/rl_agent.py` (PPOAgent)**
    *   **What happens:** The master RL algorithm. It collects the states (image features), actions (denoising strengths), and rewards. Once it collects enough, it runs the PPO (Proximal Policy Optimization) math to update the Policy networks.
*   **`rl/policy.py`**
    *   **What happens:** Contains two neural networks: 
        1. **Actor**: Looks at the image and outputs an action (e.g., "Apply 0.5 denoising strength").
        2. **Critic**: Looks at the image and guesses how much reward the model will get.
*   **`rl/reward.py`**
    *   **What happens:** Evaluates the model's final prediction. It calculates SSIM (structural similarity), penalizes blurry edges, and severely penalizes "hallucinations" (predicting structures that don't exist in the ground truth). It returns a single float score (e.g., +4.2).
*   **`rl/replay_buffer.py`**
    *   **What happens:** A simple list/array that holds the history of what happened in the last batch so `rl_agent.py` can train on it.

#### PPO Action & Reward System (Visualized)
This diagram shows the explicit interaction between the Reinforcement Learning agent, the environment (Iterative Model), and the Reward function.

```mermaid
graph TD
    subgraph PPO_Agent
        state_features["State: Image Features"]
        actor_net["Actor Network (Policy)"]
        critic_net["Critic Network (Value)"]
        action_vector["Action: Denoising Strengths [0, 1]"]
        value_est["Value Estimate"]
    end

    subgraph Environment
        denoiser_step["PDEDenoiser Applies Actions"]
        iter_loop["Iterative Loop Continues"]
        final_pred["Final FW & ICVF Maps"]
    end

    subgraph Reward_System
        ssim_calc["SSIM & Edge Consistency"]
        hall_penalty["Hallucination Penalty"]
        instab_penalty["Instability Penalty"]
        final_reward["Scalar Reward Signal (R)"]
    end

    %% Agent internals
    state_features --> actor_net
    state_features --> critic_net
    actor_net --> action_vector
    critic_net --> value_est

    %% Action to environment
    action_vector -->|"Agent tweaks noise"| denoiser_step
    denoiser_step --> iter_loop
    iter_loop ===>|"After 8 loops"| final_pred

    %% Environment to Reward
    final_pred --> ssim_calc
    final_pred --> hall_penalty
    final_pred --> instab_penalty
    ssim_calc --> final_reward
    hall_penalty --> final_reward
    instab_penalty --> final_reward

    %% PPO Feedback
    final_reward -.->|"Advantage Update"| actor_net
    final_reward -.->|"Value Loss Update"| critic_net
```

### 5. Optimization & Utilities
*   **`losses/losses.py`**
    *   **What happens:** Standard PyTorch math (like Mean Absolute Error) used to train the non-RL parts of the model (ViT, decoder, etc.).
*   **`utils/visualization.py`**
    *   **What happens:** Code that draws `matplotlib` figures and saves them as PNGs to your hard drive so you can visually see the training progress.
*   **`utils/checkpointing.py`**
    *   **What happens:** Saves `best_model.pth` to your hard drive.
