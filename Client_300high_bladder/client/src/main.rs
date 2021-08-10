use std::net::{TcpStream,TcpListener};
use std::io::{Write, Read};
use byteorder::{WriteBytesExt, NetworkEndian, ReadBytesExt};
use std::error::Error;
use std::fs::File;

use aes::Aes128;
use block_modes::{BlockMode, Cbc};
use block_modes::block_padding::Pkcs7;
use hex_literal::hex;

use ndarray::{Array2, Zip};

use csv::ReaderBuilder;
use ndarray_csv::Array2Reader;
use std::io::Cursor;

use std::fs;

type Aes128Cbc = Cbc<Aes128, Pkcs7>;

fn main() -> Result<(), Box<dyn Error>> {

    println!("Hello, world!");


    let file_x = File::open("X_300high_bladder.csv")?;
    let mut reader_x = ReaderBuilder::new().has_headers(false).from_reader(file_x);
    let array_read_x: Array2<f32> = reader_x.deserialize_array2((2500, 300))?;

    let file_x_count = File::open("Count_X_300high_bladder.csv")?;
    let mut reader_x_count = ReaderBuilder::new().has_headers(false).from_reader(file_x_count);
    let array_read_x_count: Array2<f32> = reader_x_count.deserialize_array2((2500, 300))?;


    println!("array_read size is {:?}", array_read_x.len());
    println!("array_read size is {:?}", array_read_x_count.len());


    let mut wtr = Vec::new();

    Zip::from(array_read_x.genrows())
    .apply(|input|{
        Zip::from(&input)
            .apply(|i|{
                wtr.write_f32::<NetworkEndian>(*i).expect("Error sending inputs.")
            })
    });

    Zip::from(array_read_x_count.genrows())
    .apply(|input|{
        Zip::from(&input)
            .apply(|i|{
                wtr.write_f32::<NetworkEndian>(*i).expect("Error sending inputs.")
            })
    });

    println!("array write size is {:?}", wtr.len());

    let key = hex!("000102030405060708090a0b0c0d0e0f");
    let iv = hex!("f0f1f2f3f4f5f6f7f8f9fafbfcfdfeff");
    let plaintext = &wtr;
    let cipher = Aes128Cbc::new_from_slices(&key, &iv).unwrap();

    // buffer must have enough space for message+padding
    let mut buffer = vec![0u8; 6000016];
    // copy message to the buffer
    let pos = plaintext.len();
    buffer[..pos].copy_from_slice(&plaintext);
    let ciphertext = cipher.encrypt(&mut buffer, pos).unwrap();
    println!("ciphertext size is {:?}",ciphertext.len());


    let mut receive = vec![0u8; 6000016];

    let mut params = Vec::new();

    match TcpStream::connect("localhost:3333") {
        Ok(mut stream) => {
            println!("Successfully connected to server in port 3333");

            stream.write(&ciphertext).unwrap();

            println!("Sent Hello, awaiting reply...");

            stream.read(&mut receive).unwrap();


            let mut rdr = Cursor::new(&receive);

            for _i in 0..2500{

                let package_length = rdr.read_f64::<NetworkEndian>().unwrap();
                
                params.push(package_length);
            
                }

        },
        Err(e) => {
            println!("Failed to connect: {}", e);
        }
    }
    println!("Terminated.");


    let mut file = File::create("result.txt")?;
    writeln!(file, "{:?}", params);

    Ok(())
}

