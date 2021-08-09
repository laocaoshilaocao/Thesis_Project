
use std;
use autograd as ag;
use ndarray::{Array, array, Array1, Array2,arr1};
use ndarray::s;


use ag::Graph;
use ag::tensor::Variable;
use ag::optimizers::adam;
use ag::ndarray_ext::into_shared;
use ag::tensor::Constant;
use ag::rand::seq::SliceRandom;
use std::sync::{Arc, RwLock};


type Tensor<'graph> = ag::Tensor<'graph, f64>;


use crate::utils::{Dense_relu, Dense_sigmoid, Dense_DisAct, Dense_MeanAct, Dense, assign, zerodiag, create_from_index, create_from_index_rhs, max_index};

use crate::loss_functions::{zinb, nb, cal_dist, cal_latent, target_dis};




fn inputs(g: &Graph<f64>, dim: Array1<usize>) -> (Tensor, Tensor, Tensor){

    let sf_layer = g.placeholder(&[-1, 1]);
    let x = g.placeholder(&[-1, dim[0] as isize]);
    let x_count = g.placeholder(&[-1, dim[0] as isize]);
    (sf_layer, x, x_count)
}

fn get_permutation(size: usize) -> Vec<usize> {
    let mut perm: Vec<usize> = (0..size).collect();
    perm.shuffle(&mut rand::thread_rng());
    perm
}

pub fn training(x_value: Array2<f64>, x_count_value: Array2<f64>) {

    //create autoencoder

    let cell_number = 2500;
    let dim = arr1(&[300, 128, 32]); 

    let cluster_num = 4; // for test
    let cluster_num_f = 4.0; 
    let gamma = 0.001;

    let alpha = 0.001;
    let ridge_lambda = 1.0;

    let batch_size = 25isize;

    let num_batch = cell_number / batch_size as usize;


    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();

    let clusters: Arc<RwLock<ag::NdArray<f64>>> = into_shared(rng.glorot_uniform(&[cluster_num, dim[2]]));     //value should be assigned, smallest size
    
    // network variables.
    let w1: Arc<RwLock<ag::NdArray<f64>>> = into_shared(rng.glorot_uniform(&[dim[0], dim[1]]));
    let w2: Arc<RwLock<ag::NdArray<f64>>> = into_shared(rng.glorot_uniform(&[dim[1], dim[2]]));
    
    let w3: Arc<RwLock<ag::NdArray<f64>>> = into_shared(rng.glorot_uniform(&[dim[2], dim[1]]));

    let w6: Arc<RwLock<ag::NdArray<f64>>> = into_shared(rng.glorot_uniform(&[dim[1], dim[0]]));
    let w7: Arc<RwLock<ag::NdArray<f64>>> = into_shared(rng.glorot_uniform(&[dim[1], dim[0]]));
    let w8: Arc<RwLock<ag::NdArray<f64>>> = into_shared(rng.glorot_uniform(&[dim[1], dim[0]]));
    // Make a state of adam.

    //kmeans
    let m_initial: Array2<f64> = Array::ones((1,cell_number));
    let means: Arc<RwLock<ag::NdArray<f64>>> = into_shared(m_initial);


    ag::with(|g| {
        


        let clusters = g.variable(clusters.clone());            //clusters.clone());

        let w1 = g.variable(w1.clone());  // batch size, dim[1]
        let w2 = g.variable(w2.clone());
        let w3 = g.variable(w3.clone());

        let w6 = g.variable(w6.clone());
        let w7 = g.variable(w7.clone());
        let w8 = g.variable(w8.clone());

        let params = &[w1, w2, w3, w6, w7, w8];   //w6
        let param_arrays = params
        .iter()
        .map(|v| v.get_variable_array().unwrap())
        .collect::<Vec<_>>();
        let state_pretrain = adam::AdamState::new(param_arrays.as_slice());

        let params_2 = &[clusters, w1, w2, w3, w6, w7, w8];   //w6
        let param_arrays_2 = params_2
        .iter()
        .map(|v| v.get_variable_array().unwrap())
        .collect::<Vec<_>>();
        let state_total = adam::AdamState::new(param_arrays_2.as_slice());



    

        let (sf_layer, x, x_count) = inputs(g, dim);

        let z1 =  Dense_relu(x, w1);
        let latent =  Dense(z1, w2);
 
        let (num_t, latent_p_t) = cal_latent(latent, alpha, batch_size as usize);  //cell_number

        let  latent_q_t = target_dis(latent_p_t, batch_size as usize);


        let latent_p_n = latent_p_t + zerodiag(&num_t, g);
        let latent_q_n = latent_q_t + zerodiag(&num_t, g);


        let (latent_dist1_t, latent_dist2_t) = cal_dist(latent, clusters, cluster_num, batch_size as usize);

        let z3_decoder = Dense_relu(latent, w3);


        //ZINB situation
        let pi = Dense_sigmoid(z3_decoder, w6);
        let disp = Dense_DisAct(z3_decoder, w7);
        let mean = Dense_MeanAct(z3_decoder, w8);
        let output = mean * g.matmul(sf_layer, g.ones(&[1, 300])); //是直接填写的 &[1, 500]    [1, 16]


        let likelihood_loss = zinb(pi, disp, x_count, output, ridge_lambda).show_with("likelihood_loss is");


        let kmeans_loss = g.reduce_mean(g.reduce_sum(latent_dist2_t, &[1], false), &[0], false).show_with("kmeans_loss is"); 
        

        //Self_training
        let cross_entropy = g.neg(g.reduce_sum(g.reduce_sum(latent_q_n * g.ln(latent_p_n), &[1], false), &[0], false));
        let entropy = g.neg(g.reduce_sum(g.reduce_sum(latent_q_n * g.ln(latent_q_n), &[1], false), &[0], false));
        let kl_loss = (cross_entropy - entropy).show_with("KL_loss is");
            
        let total_loss : ag::Tensor<f64> =  likelihood_loss + gamma * kl_loss + alpha * kmeans_loss; 


        // instantiate an adam optimizer with default setting.

        let grads_pretain = g.grad(&[&likelihood_loss], &[w1, w2, w3, w6, w7, w8]);   //w6
        let grads_train = g.grad(&[total_loss], &[clusters, w1, w2, w3, w6, w7, w8]); 

        
        let total_update_ops_2: &[ag::Tensor<f64>] = &adam::Adam::default().compute_updates(&[clusters, w1, w2, w3, w6, w7, w8], &grads_train, &state_total, &g);

        let test_x_value: Array2<f64> = x_value;

        let test_x_count_value: Array2<f64> = x_count_value;

        let test_sf_layer_value: Array2<f64> = Array::ones((cell_number,1));

        let pretrain_update_ops: &[ag::Tensor<f64>] = &adam::Adam::default().compute_updates(&[w1, w2, w3, w6, w7, w8], &grads_pretain, &state_pretrain, &g);   //w6

        let mut i = 0;
        for _epoch in 0..5 {
            println!("pretrain iteration {:?}", i);
            for i in 0..num_batch - 1{
                let i = i as isize ;
                let test_x_value_batch = test_x_value.slice(s![i*batch_size..(i + 1)*batch_size, ..]).into_dyn();
                let test_x_count_batch = test_x_count_value.slice(s![i*batch_size..(i + 1)*batch_size, ..]).into_dyn();
                let test_sf_layer_batch = test_sf_layer_value.slice(s![i*batch_size..(i + 1)*batch_size, ..]).into_dyn();
                g.eval(pretrain_update_ops, &[x.given(test_x_value_batch.view()), x_count.given(test_x_count_batch.view()), sf_layer.given(test_sf_layer_batch.view())]);
            }

            i = i + 1;
        }

        println!("finish pre-train");


        //kmeans++ for latent space

        //generate initial centroids
        let c1 = (g.gather(latent, g.constant(array![0.0]), 0) + g.gather(latent, g.constant(array![1.0]), 0)) / 2.0;
        let c2 = (g.gather(latent, g.constant(array![3.0]), 0) + g.gather(latent, g.constant(array![6.0]), 0))/ 2.0;
        let c3 = (g.gather(latent, g.constant(array![5.0]), 0) + g.gather(latent, g.constant(array![43.0]), 0)) / 2.0;
        let c4 = (g.gather(latent, g.constant(array![51.0]), 0) + g.gather(latent, g.constant(array![2494.0]), 0)) / 2.0;

        let initial_centroids = g.concat(&[c1, c2, c3, c4], 0);
    
        let update_initial_c = assign(&clusters, &initial_centroids , g);

        g.eval(&[update_initial_c] , &[x.given(test_x_value.view()), x_count.given(test_x_count_value.view()), sf_layer.given(test_sf_layer_value.view())]);

        let centroids = clusters;

        let mut means = g.variable(means.clone());
        let points = latent;
        let centroids_num = cluster_num;
        let centroids_num_f = cluster_num_f;
        let points_num = cell_number;

       
        //Traditional Kmeans

        let points_expanded = g.tile(g.expand_dims(points, &[0]), 0, centroids_num);    
        let centroids_expanded = g.tile(g.expand_dims(centroids, &[1]), 1, points_num);          // points number

        let distances = g.reduce_sum(g.pow(points_expanded - centroids_expanded, 2.0), &[2], false);


        let assignments = g.argmin(distances, 0, false).show_with("kmeans assignment is");


        let mut i = 0.0;

        let flag_init: Array2<f64> = Array::ones((cluster_num + 1,cell_number));
        let mut flag = g.constant(flag_init);

        for _epoch in 0..centroids_num {         

        let c1 = assignments * 0.0 + i;

        let assignments_num = g.expand_dims(g.reduce_sum(g.equal(assignments, c1), &[0], false), &[0,1]);

        let where_m = g.expand_dims(g.equal(assignments, c1), &[1]);

        let gather_t = g.expand_dims(points * where_m, &[0]);

        let means_1 = g.reduce_sum(gather_t, &[1], false);

        let mean = g.div(means_1, assignments_num);


        means = g.concat(&[means, mean], 0);

        i = i+ 1.0;

        }

        flag = g.minimum(means, flag + 999.0);
    
        let indices_array = Array::linspace(1., centroids_num_f, centroids_num).into_shape((1, centroids_num)).unwrap();

        let indices = g.constant(indices_array);    

        let new_centroids_v = g.reduce_sum(g.gather(flag, indices, 0), &[0], false);


        let update_ops = assign(&centroids, &new_centroids_v , g);

        let mut j = 0;
        for _epoch in 0..3{
            println!("kmeans iteration {:?}", j);
            g.eval(&[update_ops] , &[x.given(test_x_value.view()), x_count.given(test_x_count_value.view()), sf_layer.given(test_sf_layer_value.view())]);
            j = j + 1;
        }

        println!("finish kmeans");

        let output = centroids;
        let output2 = assignments.show_with("kmeans assignment is");

        let mut k = 0;
        for epoch in 0..5 {
            println!("funetrain iteration {:?}", k);

            for i in get_permutation(num_batch){
                let i = i as isize * batch_size;
                let test_x_value_batch = test_x_value.slice(s![i..i + batch_size, ..]);
                let test_x_count_batch = test_x_count_value.slice(s![i..i + batch_size, ..]);
                let test_sf_layer_batch = test_sf_layer_value.slice(s![i..i + batch_size, ..]);
                g.eval(total_update_ops_2, &[x.given(test_x_value_batch.view()), x_count.given(test_x_count_batch.view()), sf_layer.given(test_sf_layer_batch.view())]);
            }
            k = k + 1;
        }
        println!("finish funetrain");

        output2.eval(&[x.given(test_x_value.view()), x_count.given(test_x_count_value.view()), sf_layer.given(test_sf_layer_value.view())]);

    });

}

