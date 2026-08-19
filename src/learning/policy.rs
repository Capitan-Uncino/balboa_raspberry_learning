use nalgebra::SVector;
use std::any::Any;
use std::sync::Arc;

pub const DIM_X: usize = 4;
pub const DIM_U: usize = 1;

#[derive(Clone)]
pub struct Policy {
    action_fn: Arc<dyn Fn(&SVector<f64, DIM_X>) -> f64 + Send + Sync + 'static>,
    explicit_gains: Option<[f64; 4]>,
    // Stores the opaque deep learning records so training can be resumed
    pub network_state: Option<Arc<dyn Any + Send + Sync>>,
}

impl Policy {
    pub fn new<F>(action_fn: F, explicit_gains: Option<[f64; 4]>) -> Self
    where
        F: Fn(&SVector<f64, DIM_X>) -> f64 + Send + Sync + 'static,
    {
        Self {
            action_fn: Arc::new(action_fn),
            explicit_gains,
            network_state: None,
        }
    }

    /// Builder method to inject network state for continuous training
    pub fn with_network_state(mut self, state: Arc<dyn Any + Send + Sync>) -> Self {
        self.network_state = Some(state);
        self
    }

    pub fn get_action(&self, x: &SVector<f64, DIM_X>) -> f64 {
        (self.action_fn)(x)
    }

    pub fn get_gains(&self) -> Option<[f64; 4]> {
        self.explicit_gains
    }

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
