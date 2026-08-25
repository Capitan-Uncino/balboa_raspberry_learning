use crate::learning::policy::Policy;
use burn::backend::Autodiff;
use burn::backend::NdArray;
use burn::module::{AutodiffModule, Module};
use burn::nn::{Linear, LinearConfig, Relu};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::backend::Backend;
use burn::tensor::backend::BackendTypes;
use burn::tensor::{Tensor, TensorData};
use nalgebra::SVector;
use std::io::{self, Write};
use std::sync::{Arc, Mutex};

use burn::optim::SgdConfig;

pub const ANALYTIC_LQR_POLICY: [f64; 4] = [1.3665, 15.4366, 0.4062, 1.3743];
pub const SAMPLES_PER_ITER: usize = 50000;
pub const IQL_TAU: f32 = 0.8; // Expectile threshold
pub const IQL_BETA: f32 = 1.0; // Inverse temperature for Advantage weighting
pub const GAMMA: f32 = 0.99; // MDP Discount factor
pub const LEARNING_RATE: f64 = 1e-3;

pub const Q_COST: [f64; 4] = [10.0, 100.0, 0.0, 0.1];
pub const R_COST: [f64; 1] = [3.0];
pub const DIM_X: usize = 4;
pub const DIM_U: usize = 1;

#[derive(Debug, Clone, Copy)]
pub struct StateAction {
    pub phi: f64,
    pub theta: f64,
    pub phi_dot: f64,
    pub theta_dot: f64,
    pub u: f64,
}

#[derive(Module, Debug)]
pub struct MlpNetwork<B: Backend> {
    layer1: Linear<B>,
    layer2: Linear<B>,
    output_layer: Linear<B>,
    activation: Relu,
}

pub struct IqlModelState {
    pub actor: <MlpNetwork<B> as Module<B>>::Record,
    pub critic_q1: <MlpNetwork<B> as Module<B>>::Record,
    pub critic_q2: <MlpNetwork<B> as Module<B>>::Record,
    pub target_q1: <MlpNetwork<B> as Module<B>>::Record,
    pub target_q2: <MlpNetwork<B> as Module<B>>::Record,
    pub value_v: <MlpNetwork<B> as Module<B>>::Record,
}

impl<B: Backend> MlpNetwork<B> {
    pub fn new(input_dim: usize, hidden_dim: usize, output_dim: usize, device: &B::Device) -> Self {
        Self {
            layer1: LinearConfig::new(input_dim, hidden_dim).init(device),
            layer2: LinearConfig::new(hidden_dim, hidden_dim).init(device),
            output_layer: LinearConfig::new(hidden_dim, output_dim).init(device),
            activation: Relu::new(),
        }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.layer1.forward(input);
        let x = self.activation.forward(x);
        let x = self.layer2.forward(x);
        let x = self.activation.forward(x);
        self.output_layer.forward(x)
    }
}

// --- Dynamic Loss Implementations ---
fn expectile_loss<B: Backend>(q_val: Tensor<B, 2>, v_val: Tensor<B, 2>, tau: f32) -> Tensor<B, 1> {
    let diff = q_val - v_val;
    let weight = diff
        .clone()
        .greater_equal_elem(0.0)
        .int()
        .float()
        .mul_scalar(tau)
        + diff
            .clone()
            .lower_elem(0.0)
            .int()
            .float()
            .mul_scalar(1.0 - tau);
    (weight * diff.powf_scalar(2.0)).mean()
}

type B = Autodiff<NdArray>;

fn prepare_iql_dataset(
    batch: &[StateAction],
    num_transitions: usize,
    device: &<B as BackendTypes>::Device,
) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
    let mut raw_rewards = Vec::with_capacity(num_transitions);
    let global_scale = 1000.0; // Fixed normalization based on Q and R scale

    for i in 0..num_transitions {
        let s = &batch[i];
        let cost = Q_COST[0] * s.phi.powi(2)
            + Q_COST[1] * s.theta.powi(2)
            + Q_COST[2] * s.phi_dot.powi(2)
            + Q_COST[3] * s.theta_dot.powi(2)
            + R_COST[0] * s.u.powi(2);

        // Global normalization: preserves MDP stationarity
        raw_rewards.push((-cost / global_scale) as f32);
    }

    let mut states_flat = Vec::with_capacity(num_transitions * DIM_X);
    let mut actions_flat = Vec::with_capacity(num_transitions * DIM_U);
    let mut next_states_flat = Vec::with_capacity(num_transitions * DIM_X);

    for i in 0..num_transitions {
        let curr = &batch[i];
        let next = &batch[i + 1];

        states_flat.extend_from_slice(&[
            curr.phi as f32,
            curr.theta as f32,
            curr.phi_dot as f32,
            curr.theta_dot as f32,
        ]);
        actions_flat.push(curr.u as f32);
        next_states_flat.extend_from_slice(&[
            next.phi as f32,
            next.theta as f32,
            next.phi_dot as f32,
            next.theta_dot as f32,
        ]);
    }

    let t_states = Tensor::<B, 2>::from_data(
        TensorData::new(states_flat, vec![num_transitions, DIM_X]),
        device,
    );
    let t_actions = Tensor::<B, 2>::from_data(
        TensorData::new(actions_flat, vec![num_transitions, DIM_U]),
        device,
    );
    let t_rewards = Tensor::<B, 2>::from_data(
        TensorData::new(raw_rewards, vec![num_transitions, 1]),
        device,
    );
    let t_next_states = Tensor::<B, 2>::from_data(
        TensorData::new(next_states_flat, vec![num_transitions, DIM_X]),
        device,
    );

    (t_states, t_actions, t_rewards, t_next_states)
}

// Helper to perform soft updates on a specific Linear layer record
fn blend_linear<B: Backend>(
    target: burn::nn::LinearRecord<B>,
    online: burn::nn::LinearRecord<B>,
    tau: f32,
) -> burn::nn::LinearRecord<B> {
    // Detach the online network's weights so no gradients flow back
    let online_weight = online.weight.val().detach();

    let weight = target.weight.map(|t_w| {
        // Detach the target weights, do the math, and detach the final result
        // to guarantee it is a leaf tensor.
        (t_w.detach().mul_scalar(1.0 - tau) + online_weight.mul_scalar(tau)).detach()
    });

    let bias = match (target.bias, online.bias) {
        (Some(t_b), Some(o_b)) => {
            let online_bias = o_b.val().detach();
            Some(t_b.map(|t_b_val| {
                (t_b_val.detach().mul_scalar(1.0 - tau) + online_bias.mul_scalar(tau)).detach()
            }))
        }
        _ => None,
    };

    burn::nn::LinearRecord { weight, bias }
}

fn train_iql_actor(
    t_states: Tensor<B, 2>,
    t_actions: Tensor<B, 2>,
    t_rewards: Tensor<B, 2>,
    t_next_states: Tensor<B, 2>,
    num_transitions: usize,
    device: &<B as BackendTypes>::Device,
    previous_state: Option<IqlModelState>,
) -> (MlpNetwork<B>, IqlModelState) {
    // STOCHASTIC ACTOR: Output dimension is now DIM_U * 2 (Mean and Log_Std)
    let mut actor = MlpNetwork::<B>::new(DIM_X, 16, DIM_U * 2, device);
    let mut critic_q1 = MlpNetwork::<B>::new(DIM_X + DIM_U, 16, 1, device);
    let mut critic_q2 = MlpNetwork::<B>::new(DIM_X + DIM_U, 16, 1, device);
    let mut value_v = MlpNetwork::<B>::new(DIM_X, 16, 1, device);
    let mut target_q1 = MlpNetwork::<B>::new(DIM_X + DIM_U, 16, 1, device);
    let mut target_q2 = MlpNetwork::<B>::new(DIM_X + DIM_U, 16, 1, device);

    if let Some(state) = previous_state {
        println!(">>> Restoring previous IQL networks to resume training...");
        actor = actor.load_record(state.actor);
        critic_q1 = critic_q1.load_record(state.critic_q1);
        critic_q2 = critic_q2.load_record(state.critic_q2);
        target_q1 = target_q1.load_record(state.target_q1);
        target_q2 = target_q2.load_record(state.target_q2);
        value_v = value_v.load_record(state.value_v);
    } else {
        target_q1 = critic_q1.clone();
        target_q2 = critic_q2.clone();
    }

    //let mut optim_actor = AdamConfig::new().init::<B, MlpNetwork<B>>();
    //let mut optim_q1 = AdamConfig::new().init::<B, MlpNetwork<B>>();
    //let mut optim_q2 = AdamConfig::new().init::<B, MlpNetwork<B>>();
    //let mut optim_v = AdamConfig::new().init::<B, MlpNetwork<B>>();

    let mut optim_actor = SgdConfig::new().init::<B, MlpNetwork<B>>();
    let mut optim_q1 = SgdConfig::new().init::<B, MlpNetwork<B>>();
    let mut optim_q2 = SgdConfig::new().init::<B, MlpNetwork<B>>();
    let mut optim_v = SgdConfig::new().init::<B, MlpNetwork<B>>();

    println!(">>> Starting Original Paper IQL Optimization Loop...");
    let epochs = 60;
    let batch_size = 256;
    let bar_width: usize = 40;
    let tau = 0.005; // Polyak averaging step size

    for epoch in 0..epochs {
        let mut start_idx = 0;
        while start_idx < num_transitions {
            let end_idx = usize::min(start_idx + batch_size, num_transitions);
            let current_batch_size = end_idx - start_idx;

            let b_s = t_states.clone().slice([start_idx..end_idx]);
            let b_a = t_actions.clone().slice([start_idx..end_idx]);
            let b_r = t_rewards.clone().slice([start_idx..end_idx]);
            let b_s_next = t_next_states.clone().slice([start_idx..end_idx]);
            let b_sa = Tensor::cat(vec![b_s.clone(), b_a.clone()], 1);

            // --- 2. VALUE UPDATE ---
            let t_q1_val = target_q1.forward(b_sa.clone());
            let t_q2_val = target_q2.forward(b_sa.clone());
            let target_q = Tensor::min_pair(t_q1_val, t_q2_val).detach();

            let v_val = value_v.forward(b_s.clone());
            let v_loss = expectile_loss(target_q.clone(), v_val.clone(), IQL_TAU);
            let grads_v = v_loss.backward();
            let grads_v_params = GradientsParams::from_grads(grads_v, &value_v);
            value_v = optim_v.step(LEARNING_RATE, value_v, grads_v_params);

            // --- 3. CRITIC UPDATE ---
            let v_next = value_v.forward(b_s_next);
            let q_target = (b_r + v_next.mul_scalar(GAMMA)).detach();

            let current_q1 = critic_q1.forward(b_sa.clone());
            let q1_loss = (current_q1 - q_target.clone()).powf_scalar(2.0).mean();
            let grads_q1 = q1_loss.backward();
            let grads_q1_params = GradientsParams::from_grads(grads_q1, &critic_q1);
            critic_q1 = optim_q1.step(LEARNING_RATE, critic_q1, grads_q1_params);

            let current_q2 = critic_q2.forward(b_sa);
            let q2_loss = (current_q2 - q_target.clone()).powf_scalar(2.0).mean();
            let grads_q2 = q2_loss.backward();
            let grads_q2_params = GradientsParams::from_grads(grads_q2, &critic_q2);
            critic_q2 = optim_q2.step(LEARNING_RATE, critic_q2, grads_q2_params);

            // --- 4. STOCHASTIC ACTOR UPDATE ---
            let advantage = target_q - v_val.detach();
            let weight = (advantage.mul_scalar(IQL_BETA))
                .clamp_max(10.0)
                .exp()
                .detach();

            let pi_output = actor.forward(b_s);

            // Slice into Mean and Log Std. (Assumes DIM_U is 1 for [batch_size, 2] output)
            let mu = pi_output.clone().slice([0..current_batch_size, 0..DIM_U]);
            let log_std = pi_output
                .slice([0..current_batch_size, DIM_U..(DIM_U * 2)])
                .clamp(-5.0, 2.0);
            let std = log_std.clone().exp();

            // Calculate Gaussian Negative Log-Likelihood
            let variance = std.powf_scalar(2.0);
            let diff_sq = (b_a.clone() - mu).powf_scalar(2.0);

            // log_prob = -0.5 * ( (a-mu)^2 / var + 2*log_std + log(2*pi) )
            let log_prob =
                (diff_sq / variance + log_std.mul_scalar(2.0) + 1.837877).mul_scalar(-0.5);
            let actor_loss = (weight * -log_prob).mean(); // Maximize weighted log_prob

            let grads_actor = actor_loss.backward();
            let grads_actor_params = GradientsParams::from_grads(grads_actor, &actor);
            actor = optim_actor.step(LEARNING_RATE, actor, grads_actor_params);

            // --- 5. TARGET NETWORK SOFT UPDATE (Polyak Averaging) ---
            let t_q1_rec = target_q1.clone().into_record();
            let c_q1_rec = critic_q1.clone().into_record();
            target_q1 = target_q1.load_record(MlpNetworkRecord {
                // Auto-generated by Burn
                layer1: blend_linear(t_q1_rec.layer1, c_q1_rec.layer1, tau),
                layer2: blend_linear(t_q1_rec.layer2, c_q1_rec.layer2, tau),
                output_layer: blend_linear(t_q1_rec.output_layer, c_q1_rec.output_layer, tau),
                activation: t_q1_rec.activation,
            });

            let t_q2_rec = target_q2.clone().into_record();
            let c_q2_rec = critic_q2.clone().into_record();
            target_q2 = target_q2.load_record(MlpNetworkRecord {
                layer1: blend_linear(t_q2_rec.layer1, c_q2_rec.layer1, tau),
                layer2: blend_linear(t_q2_rec.layer2, c_q2_rec.layer2, tau),
                output_layer: blend_linear(t_q2_rec.output_layer, c_q2_rec.output_layer, tau),
                activation: t_q2_rec.activation,
            });

            start_idx += batch_size;
        }

        // --- Logging ---
        let progress = (epoch + 1) as f64 / epochs as f64;
        let filled_len = (progress * bar_width as f64).round() as usize;
        let empty_len = bar_width.saturating_sub(filled_len);
        let filled_str = "=".repeat(filled_len.saturating_sub(1));
        let head_str = if filled_len > 0 { ">" } else { "" };
        let empty_str = " ".repeat(empty_len);

        println!(
            "Epoch [{:2}/{:2}] [{}{}{}] {:>5.1}%",
            epoch + 1,
            epochs,
            filled_str,
            head_str,
            empty_str,
            progress * 100.0
        );
        let _ = io::stdout().flush();
    }

    println!(">>> IQL Compilation Complete. Packing secondary networks for storage.");

    let new_state = IqlModelState {
        actor: actor.clone().into_record(),
        critic_q1: critic_q1.clone().into_record(),
        critic_q2: critic_q2.clone().into_record(),
        target_q1: target_q1.clone().into_record(),
        target_q2: target_q2.clone().into_record(),
        value_v: value_v.clone().into_record(),
    };

    (actor, new_state)
}

pub fn get_policy(batch: &[StateAction], current_policy: &Policy) -> Policy {
    let device = <B as BackendTypes>::Device::default();
    let num_transitions = batch.len() - 1;

    assert!(
        num_transitions > 0,
        "Dataset must contain at least 2 steps to construct transitions."
    );

    let (t_states, t_actions, t_rewards, t_next_states) =
        prepare_iql_dataset(batch, num_transitions, &device);

    let previous_state: Option<IqlModelState> = current_policy
        .network_state
        .as_ref()
        .and_then(|state_any| state_any.downcast_ref::<Mutex<Option<IqlModelState>>>())
        .and_then(|mutex| mutex.lock().unwrap().take());

    let (actor, new_state) = train_iql_actor(
        t_states,
        t_actions,
        t_rewards,
        t_next_states,
        num_transitions,
        &device,
        previous_state,
    );

    println!(">>> Converting Actor to pure Inference backend...");

    let infer_actor = actor.valid();
    type InferBackend = <B as AutodiffBackend>::InnerBackend;
    let infer_device = <InferBackend as BackendTypes>::Device::default();

    // Extract gains from the current policy, fallback to the global analytic LQR gains if None
    let fallback_gains = current_policy.get_gains().unwrap_or(ANALYTIC_LQR_POLICY);

    // --- NEW: Create a thread-safe variable to track the previous alpha ---
    // We initialize it to 1.0 (assuming we start by fully trusting the Neural Network)
    let prev_alpha = Arc::new(Mutex::new(0.0f64));

    Policy::new(
        move |state_vec: &SVector<f64, DIM_X>| -> f64 {
            let state_array = [
                state_vec[0] as f32,
                state_vec[1] as f32,
                state_vec[2] as f32,
                state_vec[3] as f32,
            ];

            let state_tensor = Tensor::<InferBackend, 1>::from_data(state_array, &infer_device)
                .reshape([1, DIM_X]);

            // 1. Get NN Prediction & Uncertainty
            let action_tensor = infer_actor.forward(state_tensor);
            let nn_action = action_tensor.clone().slice([0..1, 0..DIM_U]).into_scalar() as f64;

            let log_std = action_tensor
                .slice([0..1, DIM_U..(DIM_U * 2)])
                .into_scalar() as f64;
            let std = log_std.clamp(-5.0, 2.0).exp();

            // 2. Linear Action
            let linear_action = fallback_gains[0] * state_vec[0]
                + fallback_gains[1] * state_vec[1]
                + fallback_gains[2] * state_vec[2]
                + fallback_gains[3] * state_vec[3];

            // 3. Calculate Raw Blend Factor
            let safe_std = 0.05;
            let unsafe_std = 0.5;

            let mut raw_alpha = (unsafe_std - std) / (unsafe_std - safe_std);
            raw_alpha = raw_alpha.clamp(0.0, 1.0);

            let mut p_alpha_guard = prev_alpha.lock().unwrap();

            let alpha_lowpass_constant: f64 = 0.1;
            let smoothed_alpha = alpha_lowpass_constant * raw_alpha
                + (1.0 - alpha_lowpass_constant) * *p_alpha_guard;

            // Save the smoothed alpha for the next tick
            *p_alpha_guard = smoothed_alpha;

            // println!("alpha: {}", smoothed_alpha); // Uncomment to debug smooth transitions

            // 4. Return the blended action
            (smoothed_alpha * nn_action) + ((1.0 - smoothed_alpha) * linear_action)
        },
        Some(fallback_gains),
    )
    .with_network_state(Arc::new(Mutex::new(Some(new_state))))
}
