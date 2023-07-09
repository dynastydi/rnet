use ndarray::Array2;

use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

fn max_index(arr : &Array2<f32>) -> usize {
    let mut max : f32 = 0.;
    let mut max_i : usize = 0;
    for (i, o) in arr.iter().enumerate() {
        if (o > &max) { max = *o; max_i = i; }
    }
    max_i
}


fn sigmoid(arr : &Array2<f32>) -> Array2<f32> {
    1.0 / (1.0 + arr.mapv(|el| (-el).exp()))
}

fn prime(arr : &Array2<f32>) -> Array2<f32> { 
    arr * (1.0 - arr)
}

pub struct Net { 
    learning_rate : f32,
    
    h_weights : Array2<f32>,
    o_weights : Array2<f32>
}

impl Net {
    pub fn new ( input : usize, hidden : usize, output : usize, learning_rate : f32 ) -> Self {  
        Self { 
            learning_rate,
            h_weights : Array2::random((hidden, input), Uniform::new(-0.05, 0.05)),
            o_weights : Array2::random((output, hidden), Uniform::new(-0.05, 0.05))
        }
    }  
    
    pub fn train(&mut self, in_arr : Array2<f32>, targ_arr : Array2<f32>) {
        
        let h_out = sigmoid(&self.h_weights.dot(&in_arr));
        let o_out = sigmoid(&self.o_weights.dot(&h_out));

        let o_errors = &o_out - targ_arr;
        let h_errors = self.o_weights.t().dot(&o_errors) * &h_out;
        
        let o_gradient = o_errors.dot(&h_out.t());
        let h_gradient = h_errors.dot(&in_arr.t());

        self.o_weights = &self.o_weights - (o_gradient * self.learning_rate);
        self.h_weights = &self.h_weights - (h_gradient * self.learning_rate);
    
    }

    pub fn test(&mut self, in_arr : Array2<f32>, targ_arr : Array2<f32>) -> bool {

        let h_out = sigmoid(&self.h_weights.dot(&in_arr));
        let o_out = sigmoid(&self.o_weights.dot(&h_out));
        max_index(&o_out) == max_index(&targ_arr)
    
    }
}
