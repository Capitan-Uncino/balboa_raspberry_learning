use crate::learning::policy::Policy;
use burn::backend::Autodiff;
use burn::backend::NdArray;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig, Relu};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::backend::Backend;
use burn::tensor::backend::BackendTypes;
use burn::tensor::{Tensor, TensorData};
use nalgebra::{SMatrix, SVector};
use std::io::{self, Write};
pub const ANALYTIC_LQR_POLICY: [f64; 4] = [1.3665, 15.4366, 0.4062, 1.3743];
pub const SAMPLES_PER_ITER: usize = 50000;
pub const IQL_TAU: f32 = 0.7; // Expectile threshold
pub const IQL_BETA: f32 = 3.0; // Inverse temperature for Advantage weighting
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
    let mut min_r = f64::INFINITY;
    let mut max_r = f64::NEG_INFINITY;

    // First pass: Calculate exact quadratic LQR costs
    for i in 0..num_transitions {
        let s = &batch[i];
        let cost = Q_COST[0] * s.phi.powi(2)
            + Q_COST[1] * s.theta.powi(2)
            + Q_COST[2] * s.phi_dot.powi(2)
            + Q_COST[3] * s.theta_dot.powi(2)
            + R_COST[0] * s.u.powi(2);

        let r = -cost;
        if r < min_r {
            min_r = r;
        }
        if r > max_r {
            max_r = r;
        }
        raw_rewards.push(r);
    }

    // Min-Max shift to force rewards into [0.0, 1.0] range
    let r_range = if (max_r - min_r).abs() < 1e-6 {
        1.0
    } else {
        max_r - min_r
    };
    let normalized_rewards: Vec<f32> = raw_rewards
        .iter()
        .map(|&r| ((r - min_r) / r_range) as f32)
        .collect();

    // Flatten data vectors for Tensor loading
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

    // Construct master dataset Tensors
    let t_states = Tensor::<B, 2>::from_data(
        TensorData::new(states_flat, vec![num_transitions, DIM_X]),
        device,
    );
    let t_actions = Tensor::<B, 2>::from_data(
        TensorData::new(actions_flat, vec![num_transitions, DIM_U]),
        device,
    );
    let t_rewards = Tensor::<B, 2>::from_data(
        TensorData::new(normalized_rewards, vec![num_transitions, 1]),
        device,
    );
    let t_next_states = Tensor::<B, 2>::from_data(
        TensorData::new(next_states_flat, vec![num_transitions, DIM_X]),
        device,
    );

    (t_states, t_actions, t_rewards, t_next_states)
}

fn train_iql_actor(
    t_states: Tensor<B, 2>,
    t_actions: Tensor<B, 2>,
    t_rewards: Tensor<B, 2>,
    t_next_states: Tensor<B, 2>,
    num_transitions: usize,
    device: &<B as BackendTypes>::Device,
) -> MlpNetwork<B> {
    let mut actor = MlpNetwork::<B>::new(DIM_X, 32, DIM_U, device);
    let mut critic_q1 = MlpNetwork::<B>::new(DIM_X + DIM_U, 32, 1, device);
    let mut critic_q2 = MlpNetwork::<B>::new(DIM_X + DIM_U, 32, 1, device);
    let mut value_v = MlpNetwork::<B>::new(DIM_X, 32, 1, device);

    // REMOVED: target_value_v

    let mut optim_actor = AdamConfig::new().init();
    let mut optim_q1 = AdamConfig::new().init();
    let mut optim_q2 = AdamConfig::new().init();
    let mut optim_v = AdamConfig::new().init();

    println!(">>> Starting Ultra-Lean IQL Optimization Loop...");
    let epochs = 60;
    let batch_size = 256;
    let bar_width: usize = 40;

    for epoch in 0..epochs {
        let mut start_idx = 0;
        while start_idx < num_transitions {
            let end_idx = usize::min(start_idx + batch_size, num_transitions);

            let b_s = t_states.clone().slice([start_idx..end_idx]);
            let b_a = t_actions.clone().slice([start_idx..end_idx]);
            let b_r = t_rewards.clone().slice([start_idx..end_idx]);
            let b_s_next = t_next_states.clone().slice([start_idx..end_idx]);
            let b_sa = Tensor::cat(vec![b_s.clone(), b_a.clone()], 1);

            // --- 1. VALUE UPDATE ---
            let q1_val = critic_q1.forward(b_sa.clone());
            let q2_val = critic_q2.forward(b_sa.clone());

            let target_q = Tensor::min_pair(q1_val, q2_val).detach();

            let v_val = value_v.forward(b_s.clone());
            let v_loss = expectile_loss(target_q.clone(), v_val.clone(), IQL_TAU);
            let grads_v = v_loss.backward();

            let grads_v_params = GradientsParams::from_grads(grads_v, &value_v);
            value_v = optim_v.step(LEARNING_RATE, value_v, grads_v_params);

            // --- 2. CRITIC UPDATE ---
            // USING LIVE VALUE NETWORK INSTEAD OF TARGET
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

            // --- 3. ACTOR UPDATE ---
            let advantage = target_q - v_val.detach();
            let weight = (advantage.mul_scalar(IQL_BETA)).clamp_max(10.0).exp();

            let pi_actions = actor.forward(b_s);
            let actor_loss = (weight.detach() * (pi_actions - b_a).powf_scalar(2.0)).mean();
            let grads_actor = actor_loss.backward();
            let grads_actor_params = GradientsParams::from_grads(grads_actor, &actor);
            actor = optim_actor.step(LEARNING_RATE, actor, grads_actor_params);

            start_idx += batch_size;
        }

        // REMOVED: Hardware-heavy target network syncing

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

    println!(">>> IQL Compilation Complete. Secondary networks freed.");
    actor
}
pub fn get_policy(batch: &[StateAction], _current_policy: &Policy) -> Policy {
    let device = <B as BackendTypes>::Device::default();
    let num_transitions = batch.len() - 1;

    assert!(
        num_transitions > 0,
        "Dataset must contain at least 2 steps to construct transitions."
    );

    // 1. Prepare Dataset
    let (t_states, t_actions, t_rewards, t_next_states) =
        prepare_iql_dataset(batch, num_transitions, &device);

    // 2. Train and Retrieve Actor Network
    let actor = train_iql_actor(
        t_states,
        t_actions,
        t_rewards,
        t_next_states,
        num_transitions,
        &device,
    );

    // 3. Encapsulate and Return Inference Closure
    Policy::new(
        move |state_vec: &SVector<f64, DIM_X>| -> f64 {
            let state_array = [
                state_vec[0] as f32,
                state_vec[1] as f32,
                state_vec[2] as f32,
                state_vec[3] as f32,
            ];

            let state_tensor = Tensor::<B, 1>::from_data(state_array, &device).reshape([1, DIM_X]);
            let action_tensor = actor.forward(state_tensor);

            let action_vec = action_tensor.into_data().to_vec::<f32>().unwrap();
            action_vec[0] as f64
        },
        None, // Signals that no explicit linear K matrix exists
    )
}
