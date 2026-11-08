use std::time::Instant;
use serenith_tfhe::encryption::CpuBackend;

fn main() {
    let time = Instant::now();
    let backend = CpuBackend::new(64).unwrap();
    println!("setup_seconds={:.3}", time.elapsed().as_secs_f64());
    let mut a = vec![0i8; 64];
    let mut b = vec![0i8; 64];
    a[0] = 127;
    b[0] = 127;
    let time = Instant::now();
    let ea = backend.encrypt(&a).unwrap();
    let eb = backend.encrypt(&b).unwrap();
    println!("encrypt_two_vectors_seconds={:.3}", time.elapsed().as_secs_f64());
    let time = Instant::now();
    let result = backend.evaluate(&ea, &eb, 12904).unwrap();
    println!("evaluate_seconds={:.3}", time.elapsed().as_secs_f64());
    let time = Instant::now();
    let decision = backend.decrypt(&result).unwrap();
    println!("decrypt_seconds={:.3}", time.elapsed().as_secs_f64());
    println!("decision={decision} expected=true");
    assert!(decision);
}
