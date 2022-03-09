options(warn=-1)
packages <- c("tidyverse", "tidymodels", "skimr", "DataExplorer", "ggpubr", "mosaicData", "tictoc", "h2o")

libraries <- function(packages){
  for(package in packages){
    #checks if package is installed
    if(!require(package, character.only = TRUE)){
      #If package does not exist, then it will install
      install.packages(package, dependencies = TRUE)
      #Loads package
      library(package, character.only = TRUE)
    }
  }
}
libraries(packages)
tic()

X_test <- read.csv("X_test.csv")

X_test_final <- select(X_test,-1,-4,-5,-12,-17,-18,-19,-22,-23)


X_test_final[,1] <- as.factor(X_test_final[,1])
X_test_final[,2] <- as.factor(X_test_final[,2])
X_test_final[,11] <- as.factor(X_test_final[,11])
X_test_final[,12] <- as.factor(X_test_final[,12])
X_test_final[,14] <- as.factor(X_test_final[,14])
X_test_final[,15] <- as.factor(X_test_final[,15])
X_test_final[,16] <- as.factor(X_test_final[,16])
X_test_final[,17] <- as.factor(X_test_final[,17])
X_test_final[,18] <- as.factor(X_test_final[,18])

i <- 1
modelos <- c("M","MM","Combo","DCombo")
Y_test <- data.frame(matrix(0,dim(X_test_final)[1]+1,4))

h2o.init(
  nthreads = -1,
  max_mem_size = "4g"
)

datos_test <- as.h2o((X_test_final))

for(i in 1:4){
  caminito <- str_c(getwd(),"/","modelo_",modelos[i],sep = "")
  n_model <- str_c("modelo_",modelos[i])
  
  imported_model <- h2o.import_mojo(mojo_file_path = caminito,model_id = n_model)
  prediction <- h2o.predict(imported_model, datos_test)
  pred <- as.data.frame(prediction)
  Y_test[,i] <- pred[,1]
  colnames(Y_test)[i] <- modelos[i]
  
  i <- i+1
}
h2o.shutdown(prompt = FALSE)
write.csv(Y_test,file = "Y_test.csv",force(TRUE))
options(warn=0)
toc()

