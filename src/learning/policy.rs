use nalgebra::SVector;
use std::sync::Arc;

pub const DIM_X: usize = 4;
pub const DIM_U: usize = 1;

#[derive(Clone)]
pub struct Policy {
    // Arc allows us to safely share the exact same closure/NN across threads
    // without needing to deep-copy the underlying data or networks.
    action_fn: Arc<dyn Fn(&SVector<f64, DIM_X>) -> f64 + Send + Sync + 'static>,
    explicit_gains: Option<[f64; 4]>,
}

impl Policy {
    /// Creates a new Policy with an optional explicit gain array.
    pub fn new<F>(action_fn: F, explicit_gains: Option<[f64; 4]>) -> Self
    where
        F: Fn(&SVector<f64, DIM_X>) -> f64 + Send + Sync + 'static,
    {
        Self {
            action_fn: Arc::new(action_fn),
            explicit_gains,
        }
    }

    /// Evaluates the policy (forward pass) to get the control action `u`.
    pub fn get_action(&self, x: &SVector<f64, DIM_X>) -> f64 {
        (self.action_fn)(x)
    }

    /// Returns the exact linear gains if they were explicitly provided during construction.
    pub fn get_gains(&self) -> Option<[f64; 4]> {
        self.explicit_gains
    }

    /// Approximates the equivalent linear gains using finite differencing at the equilibrium state.
    pub fn get_pseudogains(&self) -> [f64; 4] {
        let eps = 1e-4;
        let mut k_elements = [0.0; 4];

        for i in 0..4 {
            let mut basis = SVector::<f64, DIM_X>::zeros();
            basis[i] = eps;

            k_elements[i] = self.get_action(&basis) / eps;
        }

        k_elements
    }
}
