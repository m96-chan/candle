//! Forward-mode automatic differentiation: Jacobian–vector products (JVP).
//!
//! [`jvp`] computes the directional derivative of an output tensor with
//! respect to a set of seed tensors, by propagating tangents *forward*
//! through the recorded op graph (the same graph used by backpropagation).
//! This is the natural tool for objectives that need `d/dt f(z + t v)` —
//! e.g. the MeanFlow / mean-flows family of one-step generative losses —
//! where reverse-mode autodiff would require a full extra backward pass
//! through the backward graph.
//!
//! ```rust
//! use candle_core::{forward_ad::jvp, Device, Tensor, Var};
//!
//! # fn main() -> candle_core::Result<()> {
//! let x = Var::new(&[1f32, 2., 3.], &Device::Cpu)?;
//! let v = Tensor::new(&[1f32, 1., 1.], &Device::Cpu)?;
//! let y = x.sqr()?; // y = x², so JVP(v) = 2 x ⊙ v
//! let dy = jvp(&y, &[(x.as_tensor(), &v)])?;
//! assert_eq!(dy.to_vec1::<f32>()?, vec![2., 4., 6.]);
//! # Ok(())
//! # }
//! ```
//!
//! ### Requirements on the graph
//!
//! Tangents flow along recorded ops. candle only records an op when one of
//! its inputs tracks the graph (it is a [`crate::Var`] or is derived from
//! one), so **seed tensors must be `Var`s** (or downstream of one).
//! Wrapping an input is cheap: `Var::from_tensor(&x)?`.
//!
//! Tensors that do not depend on any seed have a zero tangent; those
//! sub-graphs are skipped entirely, so the cost of a JVP is roughly one
//! extra forward pass over the seed-dependent part of the graph.

use crate::op::{BinaryOp, Op, ReduceOp, UnaryOp};
use crate::{bail, Result, Tensor, TensorId};
use std::collections::HashMap;

/// Tangent of a node; `None` means an identically-zero tangent.
type MaybeTangent = Option<Tensor>;

/// Computes the Jacobian–vector product of `output` with respect to the
/// given `(seed, tangent)` pairs.
///
/// Each `tangent` must have the same shape and dtype as its `seed`. Seeds
/// not present in `output`'s graph simply contribute nothing. Returns a
/// tensor with the shape of `output` (zeros if `output` does not depend on
/// any seed).
pub fn jvp(output: &Tensor, seeds: &[(&Tensor, &Tensor)]) -> Result<Tensor> {
    let mut tangents: HashMap<TensorId, MaybeTangent> = HashMap::new();
    for (seed, tangent) in seeds {
        if seed.shape() != tangent.shape() {
            bail!(
                "jvp: seed shape {:?} does not match tangent shape {:?}",
                seed.shape(),
                tangent.shape()
            )
        }
        if seed.dtype() != tangent.dtype() {
            bail!(
                "jvp: seed dtype {:?} does not match tangent dtype {:?}",
                seed.dtype(),
                tangent.dtype()
            )
        }
        tangents.insert(seed.id(), Some((*tangent).clone()));
    }

    for node in toposort(output, &tangents) {
        let tangent = node_tangent(node, &tangents)?;
        tangents.insert(node.id(), tangent);
    }

    match tangents.get(&output.id()) {
        Some(Some(t)) => Ok(t.clone()),
        _ => output.zeros_like(),
    }
}

/// Input tensors of an op, used both for graph traversal and for the
/// zero-tangent short-circuit on unsupported ops.
fn op_args(op: &Op) -> Vec<&Tensor> {
    match op {
        Op::Binary(a, b, _) | Op::Matmul(a, b) | Op::SliceScatter0(a, b, _) => vec![a, b],
        Op::WhereCond(a, b, c) => vec![a, b, c],
        Op::Cat(xs, _) => xs.iter().collect(),
        Op::Unary(a, _)
        | Op::Cmp(a, _)
        | Op::Reduce(a, _, _)
        | Op::Affine { arg: a, .. }
        | Op::ToDType(a)
        | Op::Copy(a)
        | Op::Broadcast(a)
        | Op::Narrow(a, _, _, _)
        | Op::Reshape(a)
        | Op::ToDevice(a)
        | Op::Transpose(a, _, _)
        | Op::Permute(a, _)
        | Op::Elu(a, _)
        | Op::Powf(a, _)
        | Op::CustomOp1(a, _) => vec![a],
        Op::Gather(a, ids, _) | Op::IndexSelect(a, ids, _) => vec![a, ids],
        Op::Scatter(a, ids, src, _)
        | Op::ScatterAdd(a, ids, src, _)
        | Op::IndexAdd(a, ids, src, _)
        | Op::CustomOp3(a, ids, src, _) => vec![a, ids, src],
        Op::CustomOp2(a, b, _) => vec![a, b],
        Op::Conv1D { arg, kernel, .. }
        | Op::ConvTranspose1D { arg, kernel, .. }
        | Op::Conv2D { arg, kernel, .. }
        | Op::ConvTranspose2D { arg, kernel, .. } => vec![arg, kernel],
        Op::AvgPool2D { arg, .. }
        | Op::MaxPool2D { arg, .. }
        | Op::UpsampleNearest1D { arg, .. }
        | Op::UpsampleNearest2D { arg, .. }
        | Op::UpsampleBilinear2D { arg, .. } => vec![arg],
    }
}

/// Short label for error messages on unsupported ops.
fn op_label(op: &Op) -> String {
    match op {
        Op::CustomOp1(_, o) => format!("custom op \"{}\"", o.name()),
        Op::CustomOp2(_, _, o) => format!("custom op \"{}\"", o.name()),
        Op::CustomOp3(_, _, _, o) => format!("custom op \"{}\"", o.name()),
        Op::Scatter(..) => "scatter".to_string(),
        Op::ScatterAdd(..) => "scatter-add".to_string(),
        Op::IndexAdd(..) => "index-add".to_string(),
        Op::Conv1D { .. } => "conv1d".to_string(),
        Op::ConvTranspose1D { .. } => "conv-transpose1d".to_string(),
        Op::Conv2D { .. } => "conv2d".to_string(),
        Op::ConvTranspose2D { .. } => "conv-transpose2d".to_string(),
        Op::AvgPool2D { .. } => "avg-pool2d".to_string(),
        Op::MaxPool2D { .. } => "max-pool2d".to_string(),
        Op::UpsampleNearest1D { .. } => "upsample-nearest1d".to_string(),
        Op::UpsampleNearest2D { .. } => "upsample-nearest2d".to_string(),
        Op::UpsampleBilinear2D { .. } => "upsample-bilinear2d".to_string(),
        _ => "unsupported op".to_string(),
    }
}

/// Iterative post-order traversal of `output`'s op graph, stopping at seeds
/// (already present in `tangents`) and at leaves. The returned order has
/// dependencies before dependents.
fn toposort<'a>(output: &'a Tensor, tangents: &HashMap<TensorId, MaybeTangent>) -> Vec<&'a Tensor> {
    let mut order = Vec::new();
    let mut state: HashMap<TensorId, u8> = HashMap::new(); // 1 = entered, 2 = done
    let mut stack: Vec<(&Tensor, bool)> = vec![(output, false)];
    while let Some((node, children_done)) = stack.pop() {
        if children_done {
            state.insert(node.id(), 2);
            order.push(node);
            continue;
        }
        if state.contains_key(&node.id()) || tangents.contains_key(&node.id()) {
            continue;
        }
        state.insert(node.id(), 1);
        if let Some(op) = node.op() {
            stack.push((node, true));
            for arg in op_args(op) {
                if !state.contains_key(&arg.id()) && !tangents.contains_key(&arg.id()) {
                    stack.push((arg, false));
                }
            }
        } else {
            // Leaf without a seed: zero tangent, no post-visit needed.
            state.insert(node.id(), 2);
            order.push(node);
        }
    }
    order
}

/// Computes σ(x) = 1 / (1 + e^(−x)) from basic ops.
fn sigmoid(x: &Tensor) -> Result<Tensor> {
    (x.neg()?.exp()? + 1.0)?.recip()
}

/// Given the keepdim reduced shape stored in `Op::Reduce`, recover the list
/// of reduced dimensions.
fn reduced_dim_indices(arg: &Tensor, reduced_shape: &[usize]) -> Vec<usize> {
    arg.dims()
        .iter()
        .zip(reduced_shape.iter())
        .enumerate()
        .filter_map(|(i, (&a, &r))| (r == 1 && a != 1).then_some(i))
        .collect()
}

fn node_tangent(node: &Tensor, tangents: &HashMap<TensorId, MaybeTangent>) -> Result<MaybeTangent> {
    let op = match node.op() {
        Some(op) => op,
        None => return Ok(None),
    };
    let t = |arg: &Tensor| -> MaybeTangent { tangents.get(&arg.id()).cloned().flatten() };

    let tangent = match op {
        Op::Binary(a, b, bin_op) => {
            let (ta, tb) = (t(a), t(b));
            match bin_op {
                BinaryOp::Add => match (ta, tb) {
                    (Some(ta), Some(tb)) => Some((ta + tb)?),
                    (Some(ta), None) => Some(ta),
                    (None, Some(tb)) => Some(tb),
                    (None, None) => None,
                },
                BinaryOp::Sub => match (ta, tb) {
                    (Some(ta), Some(tb)) => Some((ta - tb)?),
                    (Some(ta), None) => Some(ta),
                    (None, Some(tb)) => Some(tb.neg()?),
                    (None, None) => None,
                },
                BinaryOp::Mul => {
                    let lhs = ta.map(|ta| ta.mul(b)).transpose()?;
                    let rhs = tb.map(|tb| a.mul(&tb)).transpose()?;
                    match (lhs, rhs) {
                        (Some(l), Some(r)) => Some((l + r)?),
                        (Some(l), None) => Some(l),
                        (None, Some(r)) => Some(r),
                        (None, None) => None,
                    }
                }
                BinaryOp::Div => {
                    // d(a/b) = da / b − a db / b².
                    let lhs = ta.map(|ta| ta.div(b)).transpose()?;
                    let rhs = tb
                        .map(|tb| node.mul(&tb)?.div(b))
                        .transpose()?
                        .map(|r| r.neg())
                        .transpose()?;
                    match (lhs, rhs) {
                        (Some(l), Some(r)) => Some((l + r)?),
                        (Some(l), None) => Some(l),
                        (None, Some(r)) => Some(r),
                        (None, None) => None,
                    }
                }
                BinaryOp::Maximum | BinaryOp::Minimum => {
                    if t(a).is_none() && t(b).is_none() {
                        None
                    } else {
                        let ta = t(a).map_or_else(|| a.zeros_like(), Ok)?;
                        let tb = t(b).map_or_else(|| b.zeros_like(), Ok)?;
                        // Tangent follows whichever side the value came from
                        // (ties take `a`, matching the elementwise kernels).
                        let pick_a = match bin_op {
                            BinaryOp::Maximum => a.ge(b)?,
                            _ => a.le(b)?,
                        };
                        Some(pick_a.where_cond(&ta, &tb)?)
                    }
                }
            }
        }
        Op::Unary(a, unary_op) => match t(a) {
            None => None,
            Some(ta) => {
                let d: Tensor = match unary_op {
                    UnaryOp::Exp => node.clone(),
                    UnaryOp::Log => a.recip()?,
                    UnaryOp::Sin => a.cos()?,
                    UnaryOp::Cos => a.sin()?.neg()?,
                    UnaryOp::Abs => a.sign()?,
                    UnaryOp::Neg => return Ok(Some(ta.neg()?)),
                    UnaryOp::Recip => node.sqr()?.neg()?,
                    UnaryOp::Sqr => a.affine(2.0, 0.0)?,
                    UnaryOp::Sqrt => node.recip()?.affine(0.5, 0.0)?,
                    UnaryOp::Gelu => {
                        // gelu(x) = 0.5 x (1 + tanh(u)), u = √(2/π)(x + κx³).
                        const KAPPA: f64 = 0.044715;
                        let sqrt_2_over_pi = (2.0 / std::f64::consts::PI).sqrt();
                        let u = (a + a.powf(3.0)?.affine(KAPPA, 0.0)?)?
                            .affine(sqrt_2_over_pi, 0.0)?;
                        let th = u.tanh()?;
                        let sech2 = (1.0 - th.sqr()?)?;
                        let du = (a.sqr()?.affine(3.0 * KAPPA, 0.0)? + 1.0)?
                            .affine(sqrt_2_over_pi, 0.0)?;
                        ((th + 1.0)?.affine(0.5, 0.0)?
                            + a.affine(0.5, 0.0)?.mul(&sech2)?.mul(&du)?)?
                    }
                    UnaryOp::GeluErf => {
                        // d gelu_erf = Φ(x) + x φ(x).
                        let sqrt_half = std::f64::consts::FRAC_1_SQRT_2;
                        let phi_cdf =
                            ((a.affine(sqrt_half, 0.0)?.erf()? + 1.0)?).affine(0.5, 0.0)?;
                        let pdf_coeff = 1.0 / (2.0 * std::f64::consts::PI).sqrt();
                        let phi_pdf = a.sqr()?.affine(-0.5, 0.0)?.exp()?.affine(pdf_coeff, 0.0)?;
                        (phi_cdf + a.mul(&phi_pdf)?)?
                    }
                    UnaryOp::Erf => {
                        let c = 2.0 / std::f64::consts::PI.sqrt();
                        a.sqr()?.neg()?.exp()?.affine(c, 0.0)?
                    }
                    UnaryOp::Relu => {
                        let zeros = a.zeros_like()?;
                        a.gt(&zeros)?.to_dtype(a.dtype())?
                    }
                    UnaryOp::Silu => {
                        // d silu = σ(x) (1 + x (1 − σ(x))).
                        let s = sigmoid(a)?;
                        let one_minus_s = (1.0 - &s)?;
                        s.mul(&(a.mul(&one_minus_s)? + 1.0)?)?
                    }
                    UnaryOp::Tanh => (1.0 - node.sqr()?)?,
                    UnaryOp::Floor | UnaryOp::Ceil | UnaryOp::Round | UnaryOp::Sign => {
                        return Ok(None)
                    }
                };
                Some(ta.mul(&d)?)
            }
        },
        Op::Cmp(_, _) => None,
        Op::Reduce(a, ReduceOp::Sum, reduced_shape) => match t(a) {
            None => None,
            Some(ta) => {
                let dims = reduced_dim_indices(a, reduced_shape);
                let summed = ta.sum_keepdim(dims)?;
                Some(summed.reshape(node.shape())?)
            }
        },
        Op::Reduce(a, ReduceOp::Max | ReduceOp::Min, reduced_shape) => match t(a) {
            None => None,
            Some(ta) => {
                // Tangent of the extremum: average the tangents over the
                // positions attaining it (exact when the extremum is unique).
                let dims = reduced_dim_indices(a, reduced_shape);
                let node_kd = node.reshape(reduced_shape.clone())?;
                let mask = node_kd.broadcast_as(a.shape())?.eq(a)?.to_dtype(a.dtype())?;
                let picked = mask.mul(&ta)?.sum_keepdim(dims.clone())?;
                let count = mask.sum_keepdim(dims)?;
                Some(picked.div(&count)?.reshape(node.shape())?)
            }
        },
        Op::Reduce(_, ReduceOp::ArgMin | ReduceOp::ArgMax, _) => None,
        Op::Matmul(a, b) => {
            let lhs = t(a).map(|ta| ta.matmul(b)).transpose()?;
            let rhs = t(b).map(|tb| a.matmul(&tb)).transpose()?;
            match (lhs, rhs) {
                (Some(l), Some(r)) => Some((l + r)?),
                (Some(l), None) => Some(l),
                (None, Some(r)) => Some(r),
                (None, None) => None,
            }
        }
        Op::WhereCond(cond, a, b) => {
            if t(a).is_none() && t(b).is_none() {
                None
            } else {
                let ta = t(a).map_or_else(|| a.zeros_like(), Ok)?;
                let tb = t(b).map_or_else(|| b.zeros_like(), Ok)?;
                Some(cond.where_cond(&ta, &tb)?)
            }
        }
        Op::Cat(args, dim) => {
            if args.iter().all(|a| t(a).is_none()) {
                None
            } else {
                let parts = args
                    .iter()
                    .map(|a| t(a).map_or_else(|| a.zeros_like(), Ok))
                    .collect::<Result<Vec<_>>>()?;
                Some(Tensor::cat(&parts, *dim)?)
            }
        }
        Op::Affine { arg, mul, .. } => t(arg).map(|ta| ta.affine(*mul, 0.0)).transpose()?,
        Op::ToDType(a) => match t(a) {
            None => None,
            Some(ta) => {
                if node.dtype().is_float() {
                    Some(ta.to_dtype(node.dtype())?)
                } else {
                    None
                }
            }
        },
        Op::Copy(a) => t(a),
        Op::Broadcast(a) => t(a).map(|ta| ta.broadcast_as(node.shape())).transpose()?,
        Op::Narrow(a, dim, start, len) => {
            t(a).map(|ta| ta.narrow(*dim, *start, *len)).transpose()?
        }
        Op::Reshape(a) => t(a).map(|ta| ta.reshape(node.shape())).transpose()?,
        Op::ToDevice(a) => t(a).map(|ta| ta.to_device(node.device())).transpose()?,
        Op::Transpose(a, d1, d2) => t(a).map(|ta| ta.transpose(*d1, *d2)).transpose()?,
        Op::Permute(a, dims) => t(a).map(|ta| ta.permute(dims.clone())).transpose()?,
        Op::Elu(a, alpha) => match t(a) {
            None => None,
            Some(ta) => {
                // d elu = 1 for x > 0, α e^x = elu(x) + α otherwise.
                let zeros = a.zeros_like()?;
                let pos = a.gt(&zeros)?.to_dtype(a.dtype())?;
                let neg_d = (node + *alpha)?.mul(&(1.0 - &pos)?)?;
                Some(ta.mul(&(pos + neg_d)?)?)
            }
        },
        Op::Powf(a, e) => t(a)
            .map(|ta| -> Result<Tensor> { ta.mul(&a.powf(e - 1.0)?.affine(*e, 0.0)?) })
            .transpose()?,
        Op::Gather(a, ids, dim) => t(a).map(|ta| ta.gather(ids, *dim)).transpose()?,
        Op::IndexSelect(a, ids, dim) => t(a).map(|ta| ta.index_select(ids, *dim)).transpose()?,
        Op::SliceScatter0(a, src, offset) => {
            if t(a).is_none() && t(src).is_none() {
                None
            } else {
                let ta = t(a).map_or_else(|| a.zeros_like(), Ok)?;
                let tsrc = t(src).map_or_else(|| src.zeros_like(), Ok)?;
                Some(ta.slice_scatter0(&tsrc, *offset)?)
            }
        }
        other => {
            // Unsupported op: fine as long as no tangent flows into it.
            if op_args(other).iter().all(|a| t(a).is_none()) {
                None
            } else {
                bail!("jvp: no forward-AD rule for {}", op_label(other))
            }
        }
    };
    Ok(tangent)
}
