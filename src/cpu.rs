//! Shared CPU inference for the floating-point and integer RTen models.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use rten::{Model, ModelOptions, RunOptions, ThreadPool, op_registry};
use rten_tensor::{Layout, Tensor};
use rusto_core::{ArrayD, CoreError, InferenceSession};

#[derive(Clone)]
pub(crate) struct CpuSession {
    model: Arc<Model>,
    options: RunOptions,
    input_name: String,
    output_name: String,
}

impl CpuSession {
    pub fn open(path: &Path, pool: Arc<ThreadPool>) -> Result<Self, CoreError> {
        let operators = op_registry!(
            Add,
            AveragePool,
            BatchNormalization,
            Concat,
            Conv,
            Cast,
            DynamicQuantizeLinear,
            MatMulInteger,
            ConvTranspose,
            Div,
            Erf,
            GlobalAveragePool,
            HardSigmoid,
            Identity,
            MatMul,
            MaxPool,
            Mul,
            ReduceMean,
            Relu,
            Reshape,
            Resize,
            Sigmoid,
            Softmax,
            Squeeze,
            Transpose,
            Unsqueeze
        );
        let model = ModelOptions::with_ops(operators)
            .prepack_weights(true)
            .load_file(path)
            .map_err(model_error)?;
        if model.input_ids().len() != 1 || model.output_ids().len() != 1 {
            return Err(CoreError::Model(
                "expected a single-input, single-output OCR model".into(),
            ));
        }
        let name = |id| {
            model
                .node_info(id)
                .and_then(|info| info.name().map(str::to_owned))
                .unwrap_or_default()
        };
        let input_name = name(model.input_ids()[0]);
        let output_name = name(model.output_ids()[0]);
        Ok(Self {
            model: Arc::new(model),
            options: RunOptions::default().with_thread_pool(Some(pool)),
            input_name,
            output_name,
        })
    }
}

fn model_error(error: impl std::fmt::Display) -> CoreError {
    CoreError::Model(error.to_string())
}

impl InferenceSession for CpuSession {
    fn run(&mut self, input: ArrayD<f32>) -> Result<ArrayD<f32>, CoreError> {
        let shape = input.shape().to_vec();
        let tensor = Tensor::from_data(shape.as_slice(), input.into_raw_vec());
        let output = self
            .model
            .run_one(tensor.into(), Some(self.options.clone()))
            .map_err(model_error)?;
        let output: Tensor<f32> = output.try_into().map_err(model_error)?;
        let output = ArrayD::from_shape_vec(output.shape().to_vec(), output.into_data())?;
        Ok(output)
    }

    fn run_all(&mut self, input: ArrayD<f32>) -> Result<HashMap<String, ArrayD<f32>>, CoreError> {
        let output = self.run(input)?;
        Ok(HashMap::from([(self.output_name.clone(), output)]))
    }

    fn run_with_inputs(
        &mut self,
        mut inputs: HashMap<String, ArrayD<f32>>,
    ) -> Result<HashMap<String, ArrayD<f32>>, CoreError> {
        if inputs.len() != 1 {
            return Err(CoreError::Model("expected exactly one model input".into()));
        }
        let input = inputs
            .remove(&self.input_name)
            .ok_or_else(|| CoreError::Model(format!("missing input {}", self.input_name)))?;
        self.run_all(input)
    }

    fn try_clone(&self) -> Option<Box<dyn InferenceSession>> {
        Some(Box::new(self.clone()))
    }
}
