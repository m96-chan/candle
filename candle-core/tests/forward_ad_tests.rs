//! Tests for forward-mode AD (JVP). Analytic cases are checked exactly;
//! composite graphs are checked against central finite differences.

use candle_core::{forward_ad::jvp, DType, Device, Result, Tensor, Var, D};

fn assert_close(a: &Tensor, b: &Tensor, tol: f64) -> Result<()> {
    let diff = (a - b)?.abs()?.flatten_all()?.max(0)?.to_dtype(DType::F64)?;
    let diff = diff.to_scalar::<f64>()?;
    assert!(diff < tol, "max abs diff {diff} exceeds tolerance {tol}");
    Ok(())
}

/// Central finite-difference JVP of `f` at `x` along `v`.
fn fd_jvp(
    f: impl Fn(&Tensor) -> Result<Tensor>,
    x: &Tensor,
    v: &Tensor,
    eps: f64,
) -> Result<Tensor> {
    let plus = f(&(x + &(v * eps)?)?)?;
    let minus = f(&(x - &(v * eps)?)?)?;
    (plus - minus)? / (2.0 * eps)
}

#[test]
fn jvp_of_square_is_2xv() -> Result<()> {
    let dev = Device::Cpu;
    let x = Var::new(&[1f32, 2., 3.], &dev)?;
    let v = Tensor::new(&[0.5f32, 1., 2.], &dev)?;
    let y = x.sqr()?;
    let dy = jvp(&y, &[(x.as_tensor(), &v)])?;
    assert_eq!(dy.to_vec1::<f32>()?, vec![1., 4., 12.]);
    Ok(())
}

#[test]
fn jvp_zero_when_independent() -> Result<()> {
    let dev = Device::Cpu;
    let x = Var::new(&[1f32, 2.], &dev)?;
    let z = Var::new(&[3f32, 4.], &dev)?;
    let v = Tensor::new(&[1f32, 1.], &dev)?;
    let y = z.exp()?; // does not depend on x
    let dy = jvp(&y, &[(x.as_tensor(), &v)])?;
    assert_eq!(dy.to_vec1::<f32>()?, vec![0., 0.]);
    Ok(())
}

#[test]
fn jvp_multiple_seeds_sum() -> Result<()> {
    let dev = Device::Cpu;
    let a = Var::new(&[2f32, 3.], &dev)?;
    let b = Var::new(&[5f32, 7.], &dev)?;
    let va = Tensor::new(&[1f32, 0.], &dev)?;
    let vb = Tensor::new(&[0f32, 1.], &dev)?;
    // y = a * b, dy = va * b + a * vb = [5, 3].
    let y = a.as_tensor().mul(b.as_tensor())?;
    let dy = jvp(&y, &[(a.as_tensor(), &va), (b.as_tensor(), &vb)])?;
    assert_eq!(dy.to_vec1::<f32>()?, vec![5., 3.]);
    Ok(())
}

#[test]
fn jvp_unary_chain_matches_fd() -> Result<()> {
    let dev = Device::Cpu;
    let x0 = Tensor::rand(0.1f64, 2.0, (3, 4), &dev)?;
    let v = Tensor::rand(-1f64, 1.0, (3, 4), &dev)?;
    let f = |x: &Tensor| -> Result<Tensor> {
        let x = Var::from_tensor(x)?;
        x.silu()?.tanh()?.exp()?.sqrt()?.log()
    };
    // Rebuild through a Var for the exact JVP.
    let x = Var::from_tensor(&x0)?;
    let y = x.silu()?.tanh()?.exp()?.sqrt()?.log()?;
    let exact = jvp(&y, &[(x.as_tensor(), &v)])?;
    let approx = fd_jvp(f, &x0, &v, 1e-4)?;
    assert_close(&exact, &approx, 1e-4)
}

#[test]
fn jvp_gelu_variants_match_fd() -> Result<()> {
    let dev = Device::Cpu;
    let x0 = Tensor::rand(-2f64, 2.0, (16,), &dev)?;
    let v = Tensor::rand(-1f64, 1.0, (16,), &dev)?;
    for variant in ["gelu", "gelu_erf", "erf", "relu"] {
        let apply = |x: &Tensor| -> Result<Tensor> {
            match variant {
                "gelu" => x.gelu(),
                "gelu_erf" => x.gelu_erf(),
                "erf" => x.erf(),
                _ => x.relu(),
            }
        };
        let x = Var::from_tensor(&x0)?;
        let y = apply(x.as_tensor())?;
        let exact = jvp(&y, &[(x.as_tensor(), &v)])?;
        let approx = fd_jvp(
            |x| apply(Var::from_tensor(x)?.as_tensor()),
            &x0,
            &v,
            1e-4,
        )?;
        assert_close(&exact, &approx, 1e-3)?;
    }
    Ok(())
}

#[test]
fn jvp_softmax_composition_matches_fd() -> Result<()> {
    // softmax composed from basic ops (max reduce, exp, sum, div) — the
    // differentiable path used by attention layers during training.
    let dev = Device::Cpu;
    let softmax = |x: &Tensor| -> Result<Tensor> {
        let max = x.max_keepdim(D::Minus1)?;
        let num = x.broadcast_sub(&max)?.exp()?;
        let den = num.sum_keepdim(D::Minus1)?;
        num.broadcast_div(&den)
    };
    let x0 = Tensor::rand(-3f64, 3.0, (2, 5), &dev)?;
    let v = Tensor::rand(-1f64, 1.0, (2, 5), &dev)?;
    let x = Var::from_tensor(&x0)?;
    let y = softmax(x.as_tensor())?;
    let exact = jvp(&y, &[(x.as_tensor(), &v)])?;
    let approx = fd_jvp(|x| softmax(Var::from_tensor(x)?.as_tensor()), &x0, &v, 1e-4)?;
    assert_close(&exact, &approx, 1e-4)
}

#[test]
fn jvp_layernorm_like_composition_matches_fd() -> Result<()> {
    let dev = Device::Cpu;
    let ln = |x: &Tensor| -> Result<Tensor> {
        let d = x.dim(D::Minus1)? as f64;
        let mean = (x.sum_keepdim(D::Minus1)? / d)?;
        let x = x.broadcast_sub(&mean)?;
        let var = (x.sqr()?.sum_keepdim(D::Minus1)? / d)?;
        x.broadcast_div(&(var + 1e-5)?.sqrt()?)
    };
    let x0 = Tensor::rand(-1f64, 1.0, (3, 8), &dev)?;
    let v = Tensor::rand(-1f64, 1.0, (3, 8), &dev)?;
    let x = Var::from_tensor(&x0)?;
    let y = ln(x.as_tensor())?;
    let exact = jvp(&y, &[(x.as_tensor(), &v)])?;
    let approx = fd_jvp(|x| ln(Var::from_tensor(x)?.as_tensor()), &x0, &v, 1e-4)?;
    assert_close(&exact, &approx, 1e-4)
}

#[test]
fn jvp_mini_attention_matches_fd() -> Result<()> {
    // matmul + transpose + softmax + reshape/narrow/cat, all in one graph.
    let dev = Device::Cpu;
    let w = Tensor::rand(-0.5f64, 0.5, (1, 6, 6), &dev)?;
    let f = |x: &Tensor, w: &Tensor| -> Result<Tensor> {
        let h = x.matmul(w)?; // [b, t, 6]
        let (q, k) = (h.narrow(2, 0, 3)?, h.narrow(2, 3, 3)?);
        let logits = q.matmul(&k.transpose(1, 2)?.contiguous()?)?;
        let max = logits.max_keepdim(D::Minus1)?;
        let num = logits.broadcast_sub(&max)?.exp()?;
        let attn = num.broadcast_div(&num.sum_keepdim(D::Minus1)?)?;
        attn.matmul(&q)
    };
    let x0 = Tensor::rand(-1f64, 1.0, (1, 4, 6), &dev)?;
    let v = Tensor::rand(-1f64, 1.0, (1, 4, 6), &dev)?;
    let x = Var::from_tensor(&x0)?;
    let y = f(x.as_tensor(), &w)?;
    let exact = jvp(&y, &[(x.as_tensor(), &v)])?;
    let approx = fd_jvp(|x| f(Var::from_tensor(x)?.as_tensor(), &w), &x0, &v, 1e-4)?;
    assert_close(&exact, &approx, 1e-4)
}

#[test]
fn jvp_where_cond_and_cat_match_fd() -> Result<()> {
    let dev = Device::Cpu;
    let f = |x: &Tensor| -> Result<Tensor> {
        let zeros = x.zeros_like()?;
        let cond = x.gt(&zeros)?;
        let picked = cond.where_cond(&x.sqr()?, &x.neg()?)?;
        Tensor::cat(&[&picked, &x.tanh()?], 1)
    };
    let x0 = Tensor::rand(-1f64, 1.0, (2, 3), &dev)?;
    let v = Tensor::rand(-1f64, 1.0, (2, 3), &dev)?;
    let x = Var::from_tensor(&x0)?;
    let y = f(x.as_tensor())?;
    let exact = jvp(&y, &[(x.as_tensor(), &v)])?;
    let approx = fd_jvp(|x| f(Var::from_tensor(x)?.as_tensor()), &x0, &v, 1e-4)?;
    assert_close(&exact, &approx, 1e-4)
}

#[test]
fn jvp_scalar_time_seed_through_broadcast() -> Result<()> {
    // Mimics mean-flows: a scalar-per-batch timestep enters via reshape +
    // broadcast; the JVP is taken along dt = 1.
    let dev = Device::Cpu;
    let x = Tensor::rand(-1f64, 1.0, (2, 4), &dev)?;
    let f = |t: &Tensor| -> Result<Tensor> {
        let t3 = t.reshape((2, 1))?;
        x.broadcast_mul(&t3)?.sin()
    };
    let t0 = Tensor::rand(0f64, 1.0, (2,), &dev)?;
    let ones = Tensor::ones((2,), DType::F64, &dev)?;
    let t = Var::from_tensor(&t0)?;
    let y = f(t.as_tensor())?;
    let exact = jvp(&y, &[(t.as_tensor(), &ones)])?;
    let approx = fd_jvp(|t| f(Var::from_tensor(t)?.as_tensor()), &t0, &ones, 1e-5)?;
    assert_close(&exact, &approx, 1e-5)
}

#[test]
fn jvp_shape_mismatch_errors() -> Result<()> {
    let dev = Device::Cpu;
    let x = Var::new(&[1f32, 2.], &dev)?;
    let v = Tensor::new(&[1f32, 1., 1.], &dev)?;
    let y = x.sqr()?;
    assert!(jvp(&y, &[(x.as_tensor(), &v)]).is_err());
    Ok(())
}
