library(rhdf5)
library(igraph)
library(SIMLR)
library(cidr)
library(SOUP)
library(RaceID)

require(rhdf5)
require(igraph)
require(SIMLR)
require(cidr)
require(SOUP)
require(RaceID)

read_clean <- function(data) {  
  if (length(dim(data)) == 1) {
    data <- as.vector(data)
  } else if (length(dim(data)) == 2) {
    data <- as.matrix(data)
  }
  data
}


trans_label<-function(cell_type){
  all_type<-unique(cell_type)
  label<-rep(1,length(cell_type))
  for(j in 1:length(all_type)){
    label[which(cell_type==all_type[j])]=j
  }
  return (label)
}

read_expr_mat_copy = function(file){
  fileinfor = H5Fopen(file)
  exprs_handle = H5Oopen(fileinfor, "exprs")
  if (H5Iget_type(exprs_handle) == "H5I_GROUP"){
    mat = new("dgCMatrix", x = read_clean(h5read(exprs_handle, "data")),
              i = read_clean(h5read(exprs_handle, "indices")),
              p = read_clean(h5read(exprs_handle, "indptr")),
              Dim = rev(read_clean(h5read(exprs_handle, "shape"))))
  }else if (H5Iget_type(exprs_handle) == "H5I_DATASET"){
    mat = read_clean(H5Dread(exprs_handle))  ##gene * cell
  }
  obs = H5Oopen(fileinfor, "obs")
  cell_type = h5read(obs, "cell_type1")
  cell_type = as.vector(cell_type)    
  
  label<-trans_label(cell_type) 
  return(list(data_matrix = mat, cell_type = cell_type, cell_label = label))
}



data_path = paste0("./dataset/Quake_10x_Limb_Muscle/data.h5")

datainfor = read_expr_mat_copy(data_path)
datacount = as.matrix(datainfor$data_matrix)
cell_type = datainfor$cell_type
cell_label = datainfor$cell_label
rownames(datacount)<-seq(nrow(datacount))
colnames(datacount)<-seq(ncol(datacount))



######## find CIDR cluster  ######  

CIDR_cluster = function(data, label){
  sc_cidr = scDataConstructor(data)
  sc_cidr = determineDropoutCandidates(sc_cidr)
  sc_cidr = wThreshold(sc_cidr)
  sc_cidr = scDissim(sc_cidr)
  sc_cidr = scPCA(sc_cidr,plotPC = FALSE)
  sc_cidr = nPC(sc_cidr)
  sc_cidr = scCluster(sc_cidr, nCluster = max(label) - min(label) + 1)
  nmi = compare(label, sc_cidr@clusters, method = "nmi")
  ari = compare(label, sc_cidr@clusters, method = "adjusted.rand")
  return(c(ari,nmi))
}




######## find SIMLR cluster  ######  

SIMLR_cluster_large = function(data, label){
  res_large_scale = SIMLR_Large_Scale(X = data, c = length(unique(label)),normalize = TRUE)
  nmi = compare(label, res_large_scale$y$cluster, method = "nmi")
  ari = compare(label, res_large_scale$y$cluster, method = "adjusted.rand")
  return(c(ari,nmi))
}




######## find RaceID cluster  ###### 

datascale<-function(x){
  if(x>10000) {
    return(20)
  }else if(2000<x& x<=10000) {
    return(30)
  }else  {return(50)}
}
RaceID_cluster<-function(data,label){
  
  scale<-datascale(ncol(data))
  sc <- SCseq(data)
  sc <- filterdata(sc,mintotal = 1000)
  
  sc <- compdist(sc,metric="pearson")
  
  sc<-clustexp(sc,cln=(max(label)-min(label)+1),sat=FALSE,bootnr=scale,FUNcluster = "kmeans")
  nmi<-compare(as.numeric(sc@cluster$kpart),label,method="nmi")
  ari<-compare(as.numeric(sc@cluster$kpart),label,method="adjusted.rand")
  
  return(c(ari,nmi))  
}  


#####   find result  of four methods     #####
print("begin CIDR cluster")
cidr_list = CIDR_cluster(datacount, cell_label)
print(cidr_list)
print("finish CIDR cluster")


print("begin SIMLR cluster")
simlr_list = SIMLR_cluster_large(datacount, cell_label)
print(simlr_list)
print("finish SIMLR cluster")

print("begin RaceID cluster")
race_list = RaceID_cluster(datacount, cell_label)
print(race_list)
print("finish RaceID cluster")

