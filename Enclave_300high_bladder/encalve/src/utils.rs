use autograd as ag;
use ag::tensor::Input;

use statrs::function::gamma::ln_gamma as lgamma_func;
use statrs::function::gamma::digamma as digamma_func;

use ndarray::ArrayD;
use ndarray::IxDyn;
use ndarray::Array;

type Tensor<'graph> = ag::Tensor<'graph, f64>;


pub fn Dense_relu<'g>(x: Tensor<'g>, w: Tensor<'g>) -> Tensor<'g> {
    let g = x.graph();
    let y1 = g.matmul(x, w);
    g.relu(y1)
}

pub fn Dense_sigmoid<'g>(x: Tensor<'g>, w: Tensor<'g>) -> Tensor<'g> {
    let g = x.graph();
    let y1 = g.matmul(x, w);
    g.sigmoid(y1)
}

pub fn Dense_DisAct<'g>(x: Tensor<'g>, w: Tensor<'g>) -> Tensor<'g> {
    let g = x.graph();
    let y1 = g.matmul(x, w);
    g.clip(g.softplus(y1), 1e-4, 1e4)
}

pub fn Dense_MeanAct<'g>(x: Tensor<'g>, w: Tensor<'g>) -> Tensor<'g> {
    let g = x.graph();
    let y1 = g.matmul(x, w);
    g.clip(g.exp(y1), 1e-5, 1e6)
}

pub fn Dense<'g>(x: Tensor<'g>, w: Tensor<'g>) -> Tensor<'g> {
    let g = x.graph();
    g.matmul(x, w)
}

pub fn Where_less<'g>(x: Tensor<'g>, y: Tensor<'g>, z: Tensor<'g>) -> Tensor<'g> {
    let g = x.graph();
    let v = 1e-8;
    let x_v = x * 0.0 + 1e-8;
    let less_matrix = g.lesser(x, x_v);
    let larger_matrix = g.lesser(x_v, x);

    y * less_matrix + z * larger_matrix
}


pub fn nan2inf<'g>(x: Tensor<'g>) -> Tensor<'g> {
    let g = x.graph();
    let v = zeros_from_tensor(&x, g) + 9999.;    // approximation of inf
    g.minimum(x, v)
}

//zeros from 2D tensor
struct Zeros_from_tensor;

impl ag::op::Op<f64> for Zeros_from_tensor {
    fn compute(
        &self,
        ctx: &mut ag::op::ComputeContext<f64>,
    ) {
        //ctx.input_mut(0).assign(&ctx.input(1));
        
        let y = &ctx.input(0).mapv(move |a| a);
        let shape = y.shape();
        //let arr1 = Array::linspace(0., y[0] - 1.0, value);

        let out = ArrayD::<f64>::zeros(IxDyn(&[shape[0], shape[1]]));
        ctx.append_output(out);
    }

    fn grad(&self, ctx: &mut ag::op::GradientContext<f64>) {
        ctx.append_input_grad(None);
    }
}


pub fn zeros_from_tensor<'graph>(x: &Tensor<'graph>, g: &'graph ag::Graph<f64>)
-> Tensor<'graph> {
    ag::Tensor::builder()
           .set_inputs(&[Input::new_mut(x)])
           .build(g, Zeros_from_tensor)
}



struct Digamma;

impl ag::op::Op<f64> for Digamma {
    fn compute(
        &self,
        ctx: &mut ag::op::ComputeContext<f64>,
    ) {
        let x: &ag::NdArrayView<_> = &ctx.input(0);
        let y = x.mapv(move |a| digamma_func(a));
        ctx.append_output(y);
    }

    fn grad(&self, ctx: &mut ag::op::GradientContext<f64>) {
        // No problem with empty implementation this time.
        ctx.append_input_grad(None);
    }
}


// helper
fn digamma<'graph>(x: &Tensor<'graph>, g: &'graph ag::Graph<f64>)
-> Tensor<'graph> {
    ag::Tensor::builder()
           .set_inputs(&[Input::new(x)])
           .build(g, Digamma)
}
struct Lgamma;

impl ag::op::Op<f64> for Lgamma {
    fn compute(
        &self,
        ctx: &mut ag::op::ComputeContext<f64>,
    ) {
        let x: &ag::NdArrayView<_> = &ctx.input(0);
        let y = x.mapv(move |a| lgamma_func(a));
        ctx.append_output(y);
    }

    fn grad(&self, ctx: &mut ag::op::GradientContext<f64>) {

        let x = ctx.input(0);
        let gy = ctx.output_grad();
        let y = ctx.output();
        let gx = gy * digamma(&x, ctx.graph());
        ctx.append_input_grad(Some(gx));

    }
}

pub fn lgamma<'graph>(x: &Tensor<'graph>, g: &'graph autograd::Graph<f64>)
-> Tensor<'graph> {
    ag::Tensor::builder()
           .set_inputs(&[Input::new(x)])
           .build(g, Lgamma)
}



//helper
struct Pow_h;

impl ag::op::Op<f64> for Pow_h {
    fn compute(
        &self,
        ctx: &mut ag::op::ComputeContext<f64>,
    ) {
        let x = &ctx.input(0);
        let y: &ag::NdArrayView<_> = &ctx.input(1);
        let mut out = x.mapv(move |a| a);
        for ((mut o, a), b) in out.iter_mut().zip(x.iter()).zip(y.iter()) {
            *o = f64::powf(*a,*b - 1.0);
        }
        ctx.append_output(out);
    }

    fn grad(&self, ctx: &mut ag::op::GradientContext<f64>) {
        ctx.append_input_grad(None);
    }
}

// helper
fn pow_h<'graph>(x: &Tensor<'graph>, y: &Tensor<'graph>, g: &'graph ag::Graph<f64>)
-> Tensor<'graph> {
    ag::Tensor::builder()
           .set_inputs(&[Input::new(x), Input::new(y)])
           .build(g, Pow_h)
}


struct Pow_e;

impl ag::op::Op<f64> for Pow_e {
    fn compute(
        &self,
        ctx: &mut ag::op::ComputeContext<f64>,
    ) {
        let x = &ctx.input(0);
        let y: &ag::NdArrayView<_> = &ctx.input(1);
        // Use `ndarray::Array::mapv` for element-wise computation.
        let mut out = x.mapv(move |a| a);
        for ((mut o, a), b) in out.iter_mut().zip(x.iter()).zip(y.iter()) {
            *o = f64::powf(*a,*b);
        }
        ctx.append_output(out);
    }

    fn grad(&self, ctx: &mut ag::op::GradientContext<f64>) {
        let x = ctx.input(0);
        let y = ctx.input(1);
        let z = ctx.output();
        let grad = ctx.output_grad();

        let gx = grad * y * pow_h(&x, &y, ctx.graph());
        let gy = grad * z * ctx.graph().ln(x);
        ctx.append_input_grad(Some(gx));
        ctx.append_input_grad(Some(gy));
    }
}

pub fn pow_e<'graph>(x: &Tensor<'graph>, y: &Tensor<'graph>, g: &'graph autograd::Graph<f64>)
-> Tensor<'graph> {
    ag::Tensor::builder()
           .set_inputs(&[Input::new(x), Input::new(y)])
           .build(g, Pow_e)
}




struct Assign;

impl ag::op::Op<f64> for Assign {
    fn compute(
        &self,
        ctx: &mut ag::op::ComputeContext<f64>,
    ) {
        ctx.input_mut(0).assign(&ctx.input(1));
        ctx.append_empty_output();
    }

    fn grad(&self, ctx: &mut ag::op::GradientContext<f64>) {
        ctx.append_input_grad(None);
    }
}


pub fn assign<'graph>(x: &Tensor<'graph>,  y: &Tensor<'graph>, g: &'graph ag::Graph<f64>)
-> Tensor<'graph> {
    ag::Tensor::builder()
           .set_inputs(&[Input::new_mut(x), Input::new(y)])
           .build(g, Assign)
}




struct Create_from_index;

impl ag::op::Op<f64> for Create_from_index {
    fn compute(
        &self,
        ctx: &mut ag::op::ComputeContext<f64>,
    ) {
        //ctx.input_mut(0).assign(&ctx.input(1));
        
        let y = &ctx.input(0).mapv(move |a| a);
        let value = y[0] as usize;
        let arr1 = Array::linspace(0., y[0] - 1.0, value);
        let mut out = ArrayD::<f64>::zeros(IxDyn(&[1, value]));
        for (o, b) in out.iter_mut().zip(arr1.iter()) {
            *o = *o + b;
        }
        ctx.append_output(out);
    }

    fn grad(&self, ctx: &mut ag::op::GradientContext<f64>) {
        ctx.append_input_grad(None);
    }
}


pub fn create_from_index<'graph>(x: &Tensor<'graph>, g: &'graph ag::Graph<f64>)
-> Tensor<'graph> {
    ag::Tensor::builder()
           .set_inputs(&[Input::new_mut(x)])
           .build(g, Create_from_index)
}


struct Create_from_index_rhs;

impl ag::op::Op<f64> for Create_from_index_rhs {
    fn compute(
        &self,
        ctx: &mut ag::op::ComputeContext<f64>,
    ) {
        //ctx.input_mut(0).assign(&ctx.input(1));
        
        let x = &ctx.input(0).mapv(move |a| a);
        let y = &ctx.input(1).mapv(move |a| a);
        let value_index = x[0] as usize;
        let value_total = y[0] as usize;

        let arr1 = Array::linspace(x[0] + 1.0, y[0] - 1.0, value_total -value_index - 1);
        let mut out = ArrayD::<f64>::zeros(IxDyn(&[1, value_total -value_index - 1]));

        for (o, b) in out.iter_mut().zip(arr1.iter()) {
            *o = *o + b;
        }
        ctx.append_output(out);
    }

    fn grad(&self, ctx: &mut ag::op::GradientContext<f64>) {
        ctx.append_input_grad(None);
    }
}


pub fn create_from_index_rhs<'graph>(x: &Tensor<'graph>, y: &Tensor<'graph>, g: &'graph ag::Graph<f64>)
-> Tensor<'graph> {
    ag::Tensor::builder()
           .set_inputs(&[Input::new_mut(x), Input::new_mut(y)])
           .build(g, Create_from_index_rhs)
}



struct Max_index;

impl ag::op::Op<f64> for Max_index {
    fn compute(
        &self,
        ctx: &mut ag::op::ComputeContext<f64>,
    ) {
        //ctx.input_mut(0).assign(&ctx.input(1));
        
        let x = &ctx.input(0);

        let y = &ctx.input(1).mapv(move |a| a);
        let max_value = y[0];

        //let arr1 = Array::linspace(x[0] + 1.0, y[0] - 1.0, value_total -value_index - 1);
        let mut i = 0.0;
        let mut j = 0.0;
        let mut out = ArrayD::<f64>::zeros(IxDyn(&[1]));

        for o in x.iter() {
            if *o == max_value
            {
                j = i;
            }
            i = i + 1.0;
            if i == 2500
            {
                i = 0.0;
            }
        }
        out[[0]] = j;
        ctx.append_output(out);
    }

    fn grad(&self, ctx: &mut ag::op::GradientContext<f64>) {
        ctx.append_input_grad(None);
    }
}


pub fn max_index<'graph>(x: &Tensor<'graph>, y: &Tensor<'graph>, g: &'graph ag::Graph<f64>)
-> Tensor<'graph> {
    ag::Tensor::builder()
           .set_inputs(&[Input::new_mut(x), Input::new_mut(y)])
           .build(g, Max_index)
}

struct Zerodiag;

impl<T: ag::Float> ag::op::Op<T> for Zerodiag {
    fn compute(
        &self,
        ctx: &mut ag::op::ComputeContext<T>,
    ) {
        let x: &ag::NdArrayView<_> = &ctx.input(0);
        let y = x.mapv(move |a| a);
        let y1 = y.diag();
        let n = y1.len();
        let mut arr = ArrayD::<T>::zeros(IxDyn(&[n, n]));
        arr.diag_mut().assign(&y1);
        //let big = y-arr;
        ctx.append_output(arr);
    }

    fn grad(&self, ctx: &mut ag::op::GradientContext<T>) {
        ctx.append_input_grad(None);
    }
}

pub fn zerodiag<'graph, F: ag::Float>(x: &ag::Tensor<'graph, F>, g: &'graph ag::Graph<F>)
-> ag::Tensor<'graph, F> {
    ag::Tensor::builder()
           .set_inputs(&[Input::new(x)])
           .build(g, Zerodiag)
}
