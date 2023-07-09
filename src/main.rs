#![allow(dead_code)]
#![allow(unused_variables)]

mod rnet;

use num_format::{Locale, ToFormattedString};

use ndarray::Array2;
use ndarray::arr2;

use std::error::Error;
use std::io::stdout;
use std::io::Write;

fn train(net : &mut rnet::Net) -> Result<(), Box<dyn Error>> {
    
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_path("mnist/mnist_train.csv");
    let mut total = 0;

    let now = std::time::Instant::now();
    for result in reader?.deserialize::<Vec<f32>>() {
        let line = result?;
        let target = line[0] as usize;
        let mut targets = Array2::from_shape_vec((10, 1), [0.01f32; 10].to_vec())?;
        targets[[target, 0]] = 0.99;

        let img = Array2::<f32>::from_shape_vec((784, 1), line[1..].to_vec())? / 255.0;

        net.train(img, targets);
        
        total += 1;

        print!("\r{} / 60,000 images trained in {:.2?}.        ", total.to_formatted_string(&Locale::en), now.elapsed());
        stdout().flush().unwrap();
        
    }
    print!("\n");
    Ok(())
}

fn test(net : &mut rnet::Net) -> Result<(), Box<dyn Error>> {

    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_path("mnist/mnist_test.csv");
    let mut success = 0;
    let mut total = 0;

    for result in reader?.deserialize::<Vec<f32>>() {
        let line = result?;
        let target = line[0] as usize;
        let mut targets = Array2::from_shape_vec((10, 1), [0.1f32; 10].to_vec())?;
        targets[[target, 0]] = 0.99;

        let img = Array2::from_shape_vec((784, 1), line[1..].to_vec())?;
        total += 1;
        success += net.test(img, targets) as u16;
        print!("\r{} / 10,000 images tested, {:.2?}% accuracy.", total.to_formatted_string(&Locale::en), (success as f32 / total as f32* 100f32));

    }
    Ok(())
}

fn main() {
    
    let epochs : u8 = 5;

    let mut net = rnet::Net::new ( 784, 256, 10, 0.1 );

    for e in 0..epochs {
        println!("\nepoch {} / {}:", e+1, epochs);
        train(&mut net);
        test(&mut net);
    }

}   

