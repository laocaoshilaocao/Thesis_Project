# Privacy-preserving clustering of single-cell RNA-seq data in Intel SGX

This repository is an example code for thesis project "Privacy-preserving clustering of single-cell RNA-seq data in Intel SGX"

Dataset in use is Quake_10x_Bladder

Pre_Processing code is written in python and can be run to generate x_matrix and x_count matrix with:
```
python pre_process --dataname"Quake_10x_Bladder" 
```
Client code and Enclave should be run separately.

For simulation test, client code and enclave code can be run directly with:
```
cargo run
```
For real hardware test, related intel SGX driver and hardware set-up should be finished. Moreover, fortanix-EDP should be installed. Then, enclave code can be run with: 
```
cargo run --target x86_64-fortanix-unknown-sgx
```
The clustering output will be printed in the terminal.
