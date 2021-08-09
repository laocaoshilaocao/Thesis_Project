
use autograd as ag;
use ag::tensor::Constant;
use ndarray::{array, Array2};

use crate::utils::{nan2inf, lgamma, Where_less, pow_e, zerodiag};

type Tensor<'graph> = ag::Tensor<'graph, f64>;

pub fn nb<'g>(theta: Tensor<'g>, y_true: Tensor<'g>, y_pred: Tensor<'g>) -> Tensor<'g> {
    let g = theta.graph();
    let eps = 1e-10;

    let mini_constant = theta * 0.0 + 1e6;
    let theta_1 = g.minimum(theta, mini_constant);
    let t1 = lgamma(&(theta_1 + eps), g) + lgamma(&(y_true + 1.0), g) - lgamma(&(y_true + theta_1 + eps), g);
    let t2 = (theta_1 + y_true) * g.ln(1.0 + (y_pred / (theta_1 + eps))) + (y_true * (g.ln(theta_1 + eps) - g.ln(y_pred + eps)));
    
    let final_r = t1 + t2;
    let final_r_1 = nan2inf(final_r);
    let final_r_2 = g.reduce_mean(final_r_1, &[0], false);    
    let final_r_3 = g.reduce_mean(final_r_2, &[0], false);
    
    final_r_3
}

pub fn zinb<'g>(pi: Tensor<'g>, theta: Tensor<'g>, y_true: Tensor<'g>, y_pred: Tensor<'g>, ridge_lambda: f64) -> Tensor<'g> {
    let g = pi.graph();
    let eps = 1e-10;
    let nb_case = nb(theta, y_true, y_pred)- g.ln(1.0 - pi + eps);
    
    let mini_constant = theta * 0.0 + 1e6;
    let theta_1 = g.minimum(theta, mini_constant);

    let zero_nb = pow_e(& (theta_1 / (theta_1 + y_pred + eps)), &theta_1, g); 
    let zero_case = g.neg(g.ln(pi + ((1.0 - pi) * zero_nb) + eps));

    let zero_case = nan2inf(zero_case);           

    let result = Where_less(y_true, zero_case, nb_case);
    let ridge = ridge_lambda * g.pow(pi, 2.);
    let result_n = result + ridge;
    
    let result_n_1 = g.reduce_mean(result_n, &[0], false);
    let result_n_2 = g.reduce_mean(result_n_1, &[0], false);


    result_n_2
}



pub fn cal_latent<'g>(hidden: Tensor<'g>, alpha: f64, cell_num: usize) -> (Tensor<'g>, Tensor<'g>) {

    let g = hidden.graph();

    let sum_y = g.reduce_sum(&g.pow(hidden,2.0), &[1], false);

    let num_1  = g.neg(2.0 * g.matmul(hidden, g.transpose(hidden, &[1,0])));  
    let num_2 = g.reshape(&sum_y, &[-1, 1]); 
    let num_3 = g.tile(g.reshape(&sum_y, &[1, -1]), 0, cell_num);    // cell_number

    let mut num_4 = num_1 + num_2 + num_3;
    num_4 = num_4 / alpha; 
    let num_5 = g.pow(1.0 + num_4, -(alpha + 1.0)/2.0);

    let zerodiag_num = num_5 - zerodiag(&num_5, g);

    let reduce_sum_z = g.tile(g.reshape(g.reduce_sum(zerodiag_num, &[1], false), &[1,-1]), 0, cell_num);      // cell_number

    let latent_p = g.transpose(g.transpose(zerodiag_num, &[1,0]) / reduce_sum_z, &[1,0]);

    (num_5,latent_p)

}

pub fn target_dis<'g>(latent_p: Tensor<'g>, cell_num: usize) -> Tensor<'g> {
    let g = latent_p.graph();
    let lat_redu_s = g.reduce_sum(latent_p, &[1], false);
    let latent_p_sum = g.tile(g.reshape(lat_redu_s, &[1,-1]), 0, cell_num);
    let latent_q = g.transpose(g.transpose(g.pow(latent_p,2.0), &[1,0]) / latent_p_sum, &[1,0]);

    let latent_q_sum = g.tile(g.reshape(g.reduce_sum(latent_q, &[1], false), &[1,-1]), 0, cell_num);
    g.transpose(g.transpose(latent_q, &[1,0]) / latent_q_sum, &[1,0])
    
}

pub fn cal_dist<'g>(hidden: Tensor<'g>, clusters: Tensor<'g>, clus_num: usize, cell_num: usize) -> (Tensor<'g>, Tensor<'g>) {
    let g = hidden.graph();
    let hidden_expand = g.tile(g.expand_dims(hidden, &[1]), 1, clus_num); // tile cluster number times
    let clusters_expand = g.tile(g.expand_dims(clusters, &[0]), 0, cell_num); // tile batch size times


    let dist1 = g.reduce_sum(&g.pow(hidden_expand - clusters_expand , 2.0) , &[2], false);

    let temp_dist1 = dist1 - g.reshape(g.reduce_min(dist1, &[1], false), &[-1,1]);
    let q  = g.exp(g.neg(temp_dist1));

    let q_sum = g.tile(g.reshape(g.reduce_sum(q, &[1], false), &[1,-1]),0, clus_num);    //// tile cluster number times

    let q1 = g.transpose(g.transpose(q, &[1,0]) / q_sum, &[1,0]);
    let q2 = g.pow(q1, 2.0);

    let q_sum_2 = g.tile(g.reshape(g.reduce_sum(q2, &[1], false), &[1,-1]),0, clus_num);     // tile cluster number times

    let q3 = g.transpose(g.transpose(q2, &[1,0]) / q_sum_2, &[1,0]);

    let dist2 = dist1 * q3;
    
    (dist1, dist2)
}



