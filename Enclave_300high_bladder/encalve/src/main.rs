use aes::Aes128;
use block_modes::{BlockMode, Cbc};
use block_modes::block_padding::Pkcs7;
use hex_literal::hex;

use byteorder::{ReadBytesExt, WriteBytesExt};
use byteorder::NetworkEndian;

use std::net::{TcpListener, TcpStream};
use std::io::{Read, Write};

use std::io::Cursor;

use ndarray::{Array2, Zip};

type Aes128Cbc = Cbc<Aes128, Pkcs7>;


use autograd as ag;
use ag::tensor::Constant;
use ag::tensor::Input;
use ndarray::array;
use ndarray::ArrayD;
use ndarray::IxDyn;
use ndarray::Array;
type Tensor<'graph> = ag::Tensor<'graph, f32>;

//extern crate loss_functions;
mod utils;

mod loss_functions;
use crate::loss_functions::{nb, zinb, cal_dist, cal_latent, target_dis};

mod training;
use crate::training::training;



fn main() {

    let listener = TcpListener::bind("0.0.0.0:3333").unwrap();
    
    println!("Server listening on port 3333");
    for stream in listener.incoming() {
        match stream {
            Ok(mut stream) => {
                println!("New connection: {}", stream.peer_addr().unwrap());

                let mut buffer = vec![0u8; 6000016];

                let key = hex!("000102030405060708090a0b0c0d0e0f");
                let iv = hex!("f0f1f2f3f4f5f6f7f8f9fafbfcfdfeff");
            
                let cipher = Aes128Cbc::new_from_slices(&key, &iv).unwrap();
                
                stream.read_exact(&mut buffer);
            
                let mut buf = buffer.to_vec();
            
                let decrypted_ciphertext = cipher.decrypt(&mut buf).unwrap();
            
                let mut rdr = Cursor::new(&decrypted_ciphertext);
            
                let mut params = Vec::new();
            
                for _i in 0..1500000{
            
                let package_length = rdr.read_f32::<NetworkEndian>().unwrap();
                
                params.push(package_length);
            
                }
            
                let (inputs_1eft, inputs_right) = params.split_at(750000);
            
                let inputs_x = Array2::<f32>::from_shape_vec(
                    (2500, 300),
                    inputs_1eft.to_vec()
                    ).unwrap();
            
                let inputs_countx = Array2::<f32>::from_shape_vec(
                    (2500, 300),
                    inputs_right.to_vec()
                    ).unwrap();
                
                
                let x_f64 = inputs_x.mapv(move |a| a as f64);
                let x_count_f64 = inputs_countx.mapv(move |a| a as f64);
                let result = training(x_f64, x_count_f64);


                let mut wtr = Vec::new();
                Zip::from(result.genrows())
                .apply(|input|{
                    Zip::from(&input)
                        .apply(|i|{
                            wtr.write_f64::<NetworkEndian>(*i).expect("Error sending inputs.")
                        })
                });
                println!("array write size is {:?}", wtr.len());

                println!("result is {:?}", result);

                stream.write(&wtr).unwrap();
                
                break;

            }
            Err(e) => {
                println!("Error: {}", e);
                /* connection failed */
            }
        }
    }
\
    //drop(listener);


}



