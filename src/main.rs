use std::process::Command;

use implicit::integrator::integrate;
use implicit::integrator::schemes::*;
use implicit::vector_space::VectorSpace;

pub fn exp_test(dt: f32) {
    let sgn = |x: f32| -x;
    let x = [1.0];
    let info = |(_ts, x, cost, count_tot, max_count, steps, t): (
        Vec<f32>,
        Vec<f32>,
        usize,
        usize,
        usize,
        usize,
        f32,
    ),
                f: &dyn Fn(f32, [f32; 1]) -> [f32; 1]| {
        let x = x[x.len() - 1];
        let sol = f(t, [t; 1])[0].exp();
        let err = (sol - x) / sol;

        println!(
        "    steps: {steps:>5}, cost: {cost:>5}, iter: {count_tot:>5}, max: {max_count:>5}, x: {:+.1e}, err: {:+.1e}",
        x,
        err
    );
        (cost, err)
    };
    let t0 = 0.0;
    let out = |[x]: [f32; 1]| x;
    for i in -2..4 {
        let m = 10.0f32.powf(i as f32);
        println!("m: {m:e}\n");
        let f = |_t: f32, x: [f32; 1]| [sgn(m * x[0])];
        let t_max = sgn(10.0f32.powf(sgn(10.0)).ln()) / m;

        print!("GL1:       \n");
        let (c_ref, err_ref) = info(integrate(x, t0, dt, t_max, gl1(), f, out), &f);
        println!("    ratio: 1.0");
        print!("RK2:       \n");
        let (c, err) = info(integrate(x, t0, dt, t_max, rk2(), f, out), &f);
        println!(
            "    ratio: cost {:.1} | err {:.1e}",
            c as f32 / c_ref as f32,
            err / err_ref
        );
        print!("GL2:       \n");
        let (c, err) = info(integrate(x, t0, dt, t_max, gl2(), f, out), &f);
        println!(
            "    ratio: cost {:.1} | err {:.1e}",
            c as f32 / c_ref as f32,
            err / err_ref
        );
        print!("RK4:       \n");
        let (c, err) = info(integrate(x, t0, dt, t_max, rk4(), f, out), &f);
        println!(
            "    ratio: cost {:.1} | err {:.1e}",
            c as f32 / c_ref as f32,
            err / err_ref
        );
        print!("GL3:       \n");
        let (c, err) = info(integrate(x, t0, dt, t_max, gl3(), f, out), &f);
        println!(
            "    ratio: cost {:.1} | err {:.1e}",
            c as f32 / c_ref as f32,
            err / err_ref
        );
        println!("\n");
    }
}
pub fn cos_test() {
    let x = [0.15];
    let info = |(_ts, x, cost, count_tot, max_count, steps, t): (
        Vec<f32>,
        Vec<f32>,
        usize,
        usize,
        usize,
        usize,
        f32,
    )| {
        let x = x[x.len() - 1];
        // let sol = f(t, [t; 1])[0].exp();
        let sol = t.cos();
        let err = (sol - x) / sol;

        println!(
        "    steps: {steps:>5}, cost: {cost:>5}, iter: {count_tot:>5}, max: {max_count:>5}, x: {:+.1e}, err: {:+.1e}",
        x,
        err
    );
        (cost, err)
    };
    let t0 = 0.0;
    let out = |[x]: [f32; 1]| x;
    for i in -5..=0 {
        let dt = 3.0f32.powf(i as f32);
        println!("dt: {dt:e}\n");
        let f = |t: f32, x: [f32; 1]| [-1000.0 * (x[0] - t.cos())];
        // let f = |x: [f32; 1]| [sgn(m * x[0])];
        let t_max = 10.0;

        print!("GL1:       \n");
        let (c_ref, err_ref) = info(integrate(x, t0, dt, t_max, gl1(), f, out));
        println!("    ratio: 1.0");
        print!("RK2:       \n");
        let (c, err) = info(integrate(x, t0, dt, t_max, rk2(), f, out));
        println!(
            "    ratio: cost {:.1} | err {:.1e}",
            c as f32 / c_ref as f32,
            err / err_ref
        );
        print!("GL2:       \n");
        let (c, err) = info(integrate(x, t0, dt, t_max, gl2(), f, out));
        println!(
            "    ratio: cost {:.1} | err {:.1e}",
            c as f32 / c_ref as f32,
            err / err_ref
        );
        print!("RK4:       \n");
        let (c, err) = info(integrate(x, t0, dt, t_max, rk4(), f, out));
        println!(
            "    ratio: cost {:.1} | err {:.1e}",
            c as f32 / c_ref as f32,
            err / err_ref
        );
        print!("GL3:       \n");
        let (c, err) = info(integrate(x, t0, dt, t_max, gl3(), f, out));
        println!(
            "    ratio: cost {:.1} | err {:.1e}",
            c as f32 / c_ref as f32,
            err / err_ref
        );
        println!("\n");
    }
}
pub fn sin_cos_test() {
    let x = [1.0, 0.0];
    let info = |(_ts, x, cost, count_tot, max_count, steps, t): (
        Vec<f32>,
        Vec<[f32; 2]>,
        usize,
        usize,
        usize,
        usize,
        f32,
    )| {
        let [x, y] = x[x.len() - 1];
        let x = [x * x + y * y];
        let sol = [1.0];
        // let x = x[x.len() - 1];
        // let sol = [t.cos(), t.sin(), 1.0];
        let err = sol.sub(x).div(sol).norm2().sqrt();

        println!(
        "    steps: {steps:>5}, cost: {cost:>5}, iter: {count_tot:>5}, max: {max_count:>5}, err: {:+.1e}",
        err
    );
        (cost, err)
    };
    let t0 = 0.0;
    let out = |x: [f32; 2]| x;
    for i in -5..=2 {
        let dt = 3.0f32.powf(i as f32);
        println!("dt: {dt:e}\n");
        let f = |_t: f32, [x, y]: [f32; 2]| [-y, x];
        let t_max = 1e3;

        print!("GL1:       \n");
        let (c_ref, err_ref) = info(integrate(x, t0, dt, t_max, gl1(), f, out));
        println!("    ratio: 1.0");
        print!("RK2:       \n");
        let (c, err) = info(integrate(x, t0, dt, t_max, rk2(), f, out));
        println!(
            "    ratio: cost {:.1} | err {:.1e}",
            c as f32 / c_ref as f32,
            err / err_ref
        );
        print!("GL2:       \n");
        let (c, err) = info(integrate(x, t0, dt, t_max, gl2(), f, out));
        println!(
            "    ratio: cost {:.1} | err {:.1e}",
            c as f32 / c_ref as f32,
            err / err_ref
        );
        print!("RK4:       \n");
        let (c, err) = info(integrate(x, t0, dt, t_max, rk4(), f, out));
        println!(
            "    ratio: cost {:.1} | err {:.1e}",
            c as f32 / c_ref as f32,
            err / err_ref
        );
        print!("GL3:       \n");
        let (c, err) = info(integrate(x, t0, dt, t_max, gl3(), f, out));
        println!(
            "    ratio: cost {:.1} | err {:.1e}",
            c as f32 / c_ref as f32,
            err / err_ref
        );
        println!("\n");
    }
}

pub fn spring_test(tmax: f32, dt: f32, strength: f32, damping: f32, mass: f32) {
    let x0 = [0.0, 0.0, 0.75, 0.0, 2.0, 0.0];
    let t0 = 0.0;
    let l = 1.0;

    let f = |_t: f32, [x0, v0, x1, v1, x2, v2]: [f32; 6]| {
        let f01 = (((x0 - x1).abs() - l) * strength + (v1 - v0) / damping) / mass;
        let f12 = (((x1 - x2).abs() - l) * strength + (v2 - v1) / damping) / mass;
        [v0, f01, v1, -f01 + f12, v2, -f12]
    };
    let out = |[x0, _, x1, ..]: [f32; 6]| (x0 - x1).abs();
    if true {
        let (tsr, lsr, ..) = integrate(x0, t0, 1e-6, tmax, gl3(), f, out);
        let reference = tsr
            .into_iter()
            .zip(lsr.into_iter())
            .step_by(100)
            .map(|(t, l)| format!("{t} {l}"))
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write("target/ref", reference).unwrap();
    }
    let (ts, lsgl1, c1, _, m1, ..) = integrate(x0, t0, dt, tmax, gl1(), f, out);
    let (_, lsrk2, crk2, _, mrk2, ..) = integrate(x0, t0, dt, tmax, rk2(), f, out);
    let (_, lsgl2, c2, _, m2, ..) = integrate(x0, t0, dt, tmax, gl2(), f, out);
    let (_, lsgl3, c3, _, m3, ..) = integrate(x0, t0, dt, tmax, gl3(), f, out);
    println!("gl1 {c1} {m1}\nrk2 {crk2} {mrk2}\ngl2 {c2} {m2}\ngl3 {c3} {m3}");
    let vals = ts
        .into_iter()
        .zip(lsgl1.into_iter())
        .zip(lsrk2.into_iter())
        .zip(lsgl2.into_iter())
        .zip(lsgl3.into_iter())
        .map(|((((t, lgl1), lrk2), lgl2), lgl3)| format!("{t} {lgl1} {lgl2} {lgl3} {lrk2}"))
        .collect::<Vec<_>>()
        .join("\n");
    std::fs::write("target/vals", vals).unwrap();
    Command::new("gnuplot").args(["-p", "-e", "set yrange [0:2]; plot 'target/ref' u 1:2 w l t 'ref', 'target/vals' u 1:2 t 'gl1', 'target/vals' u 1:3 t 'gl2', 'target/vals' u 1:4 t 'gl3', 'target/vals' u 1:5 t 'rk2', 'target/vals' u 1:6 t 'rk4'"]).output().unwrap();
}

fn main() {
    let mass = std::env::args()
        .skip(1)
        .next()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(1e0);
    let tmax = std::env::args()
        .skip(2)
        .next()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(1.0);
    let dt = std::env::args()
        .skip(3)
        .next()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(1e-3);
    let strength = std::env::args()
        .skip(4)
        .next()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(1e6);
    let damping = std::env::args()
        .skip(5)
        .next()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(5e-4);
    spring_test(tmax, dt, strength, damping, mass);
    // exp_test(dt);
    // cos_test();
    // sin_cos_test();
}
