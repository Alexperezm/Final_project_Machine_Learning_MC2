################################################################################
#####################Final Project - Machine Learning###########################
###########################Marugán Pinos, Rocío#################################
###########################Pérez Maiztegui, Alex################################
################################################################################

#Library loading:
library(tidyverse)
library(lubridate)
library(tidymodels)
library(skimr)
library(DataExplorer)
library(ggpubr)
library(mosaicData)
library(h2o)


# Data processing:

X_train <- read.csv("X_train.csv")
Y_train <- read.csv("Y_train.csv")


#dim(X_train)
#colnames(X_train)
X_test_final <- select(X_train,-1,-4,-5,-12,-17,-18,-19,-22,-23)

#summary(X_test_final)
X_test_final[,1] <- as.factor(X_test_final[,1])
X_test_final[,2] <- as.factor(X_test_final[,2])
X_test_final[,11] <- as.factor(X_test_final[,11])
X_test_final[,12] <- as.factor(X_test_final[,12])
X_test_final[,14] <- as.factor(X_test_final[,14])
X_test_final[,15] <- as.factor(X_test_final[,15])
X_test_final[,16] <- as.factor(X_test_final[,16])
X_test_final[,17] <- as.factor(X_test_final[,17])
X_test_final[,18] <- as.factor(X_test_final[,18])
#summary(X_test_final)


Y_train[,1] <- as.factor(Y_train[,1])
Y_train[,2] <- as.factor(Y_train[,2])
Y_train[,3] <- as.factor(Y_train[,3])
Y_train[,4] <- as.factor(Y_train[,4])
#summary(Y_train)


# Data Summary:

skim(X_test_final)
skim(Y_train)


plot_missing(
  data    = X_test_final, 
  title   = "Porcentaje de valores ausentes",
  ggtheme = theme_bw(),
  theme_config = list(legend.position = "none")
)


plot_density(
  data    = X_test_final,
  ncol    = 3,
  title   = "Distribución variables continuas",
  ggtheme = theme_bw(),
  theme_config = list(
    plot.title = element_text(size = 14, face = "bold"),
    strip.text = element_text(colour = "black", size = 12, face = 2)
  )
)

plot_bar(
  data = X_test_final,
  ncol    = 3,
  title   = "Número de observaciones por grupo",
  ggtheme = theme_bw(),
  theme_config = list(
    plot.title = element_text(size = 14, face = "bold"),
    strip.text = element_text(colour = "black", size = 8, face = 2),
    legend.position = "none"
  ))



# Model_M:

M <- select(Y_train,M)
datos <- X_test_final%>%mutate(M)


# Reparto de datos en train y test
# ==============================================================================
set.seed(123)
split_inicial <- initial_split(
  data   = datos,
  prop   = 0.8,
  strata = M
)

datos_train <- training(split_inicial)
datos_test  <- testing(split_inicial)

summary(datos_train$M)
summary(datos_test$M)


# Se almacenan en un objeto `recipe` todos los pasos de preprocesado y, finalmente,
# se aplican a los datos.
transformer <- recipe(
  formula = M ~ .,
  data =  datos_train
) %>%
  step_naomit(all_predictors()) %>%
  step_nzv(all_predictors()) %>%
  step_center(all_numeric(), -all_outcomes()) %>%
  step_scale(all_numeric(), -all_outcomes()) %>%
  step_dummy(all_nominal(), -all_outcomes())

transformer


# Se entrena el objeto recipe
transformer_fit <- prep(transformer)

# Se aplican las transformaciones al conjunto de entrenamiento y de test
datos_train_prep <- bake(transformer_fit, new_data = datos_train)
datos_test_prep  <- bake(transformer_fit, new_data = datos_test)

glimpse(datos_train_prep)


# Inicialización del cluster
# ==============================================================================
h2o.init(
  nthreads = -1,
  max_mem_size = "4g"
)


# Se eliminan los datos del cluster por si ya había sido iniciado.
h2o.removeAll()
h2o.no_progress()

datos_train  <- as.h2o(datos_train_prep, key = "datos_train")
datos_test   <- as.h2o(datos_test_prep, key = "datos_test")

# Espacio de búsqueda de cada hiperparámetro
# ==============================================================================
hiperparametros <- list(
  epochs = c(50, 100, 500,1000),
  hidden = list(5, 10, 25, 50,100, c(10, 10))
)


# Búsqueda por validación cruzada
# ==============================================================================
variable_respuesta <- 'M'
predictores <- setdiff(colnames(datos_train), variable_respuesta)

grid <- h2o.grid(
  algorithm    = "deeplearning",
  activation   = "Rectifier",
  x            = predictores,
  y            = variable_respuesta,
  training_frame  = datos_train,
  nfolds       = 20, #validacion cruzada
  standardize  = FALSE,
  hyper_params = hiperparametros,
  search_criteria = list(strategy = "Cartesian"),
  seed         = 123,
  grid_id      = "grid"
)

resultados_grid <- h2o.getGrid(
  sort_by = 'rmse',
  grid_id = "grid",
  decreasing = FALSE
)
data.frame(resultados_grid@summary_table)

# Mejor modelo encontrado
# ==============================================================================
modelo_final <- h2o.getModel(resultados_grid@model_ids[[1]])

predicciones <- h2o.predict(
  object  = modelo_final,
  newdata = datos_test
)

predicciones <- predicciones %>%
  as_tibble() %>%
  mutate(valor_real = as.vector(datos_test$M))

predicciones %>% head(5)

#rmse(predicciones, truth = M, estimate = predict, na_rm = TRUE)

modelo_final@allparameters

#h2o.saveModel(modelo_final, path = getwd(), filename = "mymodel")
#saved_model <- h2o.loadModel("C:/Users/Alex/OneDrive - alumni.unav.es/Master_Metodos_Computacionales_en_Ciencias/Machine learning/FINAL/mymodel")
#my_local_model <- h2o.download_model(modelo_final, path = getwd())

mojo_destination <- h2o.save_mojo(modelo_final, path = getwd(),filename = "modelo_M")
#imported_model <- h2o.import_mojo(mojo_destination)

#h2o.predict(imported_model, datos_test)
# Se apaga el cluster H2O
h2o.shutdown(prompt = FALSE)

(sum(predicciones[,1]==predicciones[,5]))/dim(predicciones)[1]


# Modelo_MM:

MM <- select(Y_train,MM)
datos <- X_test_final%>%mutate(MM)


# Reparto de datos en train y test
# ==============================================================================
set.seed(123)
split_inicial <- initial_split(
  data   = datos,
  prop   = 0.8,
  strata = MM
)

datos_train <- training(split_inicial)
datos_test  <- testing(split_inicial)

summary(datos_train$MM)
summary(datos_test$MM)


# Se almacenan en un objeto `recipe` todos los pasos de preprocesado y, finalmente,
# se aplican a los datos.
transformer <- recipe(
  formula = MM ~ .,
  data =  datos_train
) %>%
  step_naomit(all_predictors()) %>%
  step_nzv(all_predictors()) %>%
  step_center(all_numeric(), -all_outcomes()) %>%
  step_scale(all_numeric(), -all_outcomes()) %>%
  step_dummy(all_nominal(), -all_outcomes())

transformer


# Se entrena el objeto recipe
transformer_fit <- prep(transformer)

# Se aplican las transformaciones al conjunto de entrenamiento y de test
datos_train_prep <- bake(transformer_fit, new_data = datos_train)
datos_test_prep  <- bake(transformer_fit, new_data = datos_test)

glimpse(datos_train_prep)


# Inicialización del cluster
# ==============================================================================
h2o.init(
  nthreads = -1,
  max_mem_size = "4g"
)


# Se eliminan los datos del cluster por si ya había sido iniciado.
h2o.removeAll()
h2o.no_progress()

datos_train  <- as.h2o(datos_train_prep, key = "datos_train")
datos_test   <- as.h2o(datos_test_prep, key = "datos_test")

# Espacio de búsqueda de cada hiperparámetro
# ==============================================================================
hiperparametros <- list(
  epochs = c(50, 100, 500, 1000),
  hidden = list(5, 10, 25, 50, 100, c(10, 10))
)


# Búsqueda por validación cruzada
# ==============================================================================
variable_respuesta <- 'MM'
predictores <- setdiff(colnames(datos_train), variable_respuesta)

grid <- h2o.grid(
  algorithm    = "deeplearning",
  activation   = "Rectifier",
  x            = predictores,
  y            = variable_respuesta,
  training_frame  = datos_train,
  nfolds       = 20, #validacion cruzada
  standardize  = FALSE,
  hyper_params = hiperparametros,
  search_criteria = list(strategy = "Cartesian"),
  seed         = 123,
  grid_id      = "grid"
)

resultados_grid <- h2o.getGrid(
  sort_by = 'rmse',
  grid_id = "grid",
  decreasing = FALSE
)
data.frame(resultados_grid@summary_table)

# Mejor modelo encontrado
# ==============================================================================
modelo_final <- h2o.getModel(resultados_grid@model_ids[[1]])

predicciones <- h2o.predict(
  object  = modelo_final,
  newdata = datos_test
)

predicciones <- predicciones %>%
  as_tibble() %>%
  mutate(valor_real = as.vector(datos_test$MM))

predicciones %>% head(5)

#rmse(predicciones, truth = MM, estimate = predict, na_rm = TRUE)

modelo_final@allparameters
mojo_destination <- h2o.save_mojo(modelo_final, path = getwd(),filename = "modelo_MM")

# Se apaga el cluster H2O
h2o.shutdown(prompt = FALSE)

(sum(predicciones[,1]==predicciones[,5]))/dim(predicciones)[1]


#Model_Combo:

Combo <- select(Y_train,Combo)
datos <- X_test_final%>%mutate(Combo)


# Reparto de datos en train y test
# ==============================================================================
set.seed(123)
split_inicial <- initial_split(
  data   = datos,
  prop   = 0.8,
  strata = Combo
)

datos_train <- training(split_inicial)
datos_test  <- testing(split_inicial)

summary(datos_train$Combo)
summary(datos_test$Combo)


# Se almacenan en un objeto `recipe` todos los pasos de preprocesado y, finalmente,
# se aplican a los datos.
transformer <- recipe(
  formula = Combo ~ .,
  data =  datos_train
) %>%
  step_naomit(all_predictors()) %>%
  step_nzv(all_predictors()) %>%
  step_center(all_numeric(), -all_outcomes()) %>%
  step_scale(all_numeric(), -all_outcomes()) %>%
  step_dummy(all_nominal(), -all_outcomes())

transformer


# Se entrena el objeto recipe
transformer_fit <- prep(transformer)

# Se aplican las transformaciones al conjunto de entrenamiento y de test
datos_train_prep <- bake(transformer_fit, new_data = datos_train)
datos_test_prep  <- bake(transformer_fit, new_data = datos_test)

glimpse(datos_train_prep)


# Inicialización del cluster
# ==============================================================================
h2o.init(
  nthreads = -1,
  max_mem_size = "4g"
)


# Se eliminan los datos del cluster por si ya había sido iniciado.
h2o.removeAll()
h2o.no_progress()

datos_train  <- as.h2o(datos_train_prep, key = "datos_train")
datos_test   <- as.h2o(datos_test_prep, key = "datos_test")

# Espacio de búsqueda de cada hiperparámetro
# ==============================================================================
hiperparametros <- list(
  epochs = c(50, 100, 500, 1000),
  hidden = list(5, 10, 25, 50, 100, c(10, 10))
)


# Búsqueda por validación cruzada
# ==============================================================================
variable_respuesta <- 'Combo'
predictores <- setdiff(colnames(datos_train), variable_respuesta)

grid <- h2o.grid(
  algorithm    = "deeplearning",
  activation   = "Rectifier",
  x            = predictores,
  y            = variable_respuesta,
  training_frame  = datos_train,
  nfolds       = 20, #validacion cruzada
  standardize  = FALSE,
  hyper_params = hiperparametros,
  search_criteria = list(strategy = "Cartesian"),
  seed         = 123,
  grid_id      = "grid"
)

resultados_grid <- h2o.getGrid(
  sort_by = 'rmse',
  grid_id = "grid",
  decreasing = FALSE
)
data.frame(resultados_grid@summary_table)

# Mejor modelo encontrado
# ==============================================================================
modelo_final <- h2o.getModel(resultados_grid@model_ids[[1]])

predicciones <- h2o.predict(
  object  = modelo_final,
  newdata = datos_test
)

predicciones <- predicciones %>%
  as_tibble() %>%
  mutate(valor_real = as.vector(datos_test$Combo))

predicciones %>% head(5)

#rmse(predicciones, truth = Combo, estimate = predict, na_rm = TRUE)

modelo_final@allparameters
mojo_destination <- h2o.save_mojo(modelo_final, path = getwd(),filename = "modelo_Combo")

# Se apaga el cluster H2O
h2o.shutdown(prompt = FALSE)

(sum(predicciones[,1]==predicciones[,6]))/dim(predicciones)[1]

#Model_DCombo:

DCombo <- select(Y_train,DCombo)
datos <- X_test_final%>%mutate(DCombo)


# Reparto de datos en train y test
# ==============================================================================
set.seed(123)
split_inicial <- initial_split(
  data   = datos,
  prop   = 0.8,
  strata = DCombo
)

datos_train <- training(split_inicial)
datos_test  <- testing(split_inicial)

summary(datos_train$DCombo)
summary(datos_test$DCombo)


# Se almacenan en un objeto `recipe` todos los pasos de preprocesado y, finalmente,
# se aplican a los datos.
transformer <- recipe(
  formula = DCombo ~ .,
  data =  datos_train
) %>%
  step_naomit(all_predictors()) %>%
  step_nzv(all_predictors()) %>%
  step_center(all_numeric(), -all_outcomes()) %>%
  step_scale(all_numeric(), -all_outcomes()) %>%
  step_dummy(all_nominal(), -all_outcomes())

transformer


# Se entrena el objeto recipe
transformer_fit <- prep(transformer)

# Se aplican las transformaciones al conjunto de entrenamiento y de test
datos_train_prep <- bake(transformer_fit, new_data = datos_train)
datos_test_prep  <- bake(transformer_fit, new_data = datos_test)

glimpse(datos_train_prep)


# Inicialización del cluster
# ==============================================================================
h2o.init(
  nthreads = -1,
  max_mem_size = "4g"
)


# Se eliminan los datos del cluster por si ya había sido iniciado.
h2o.removeAll()
h2o.no_progress()

datos_train  <- as.h2o(datos_train_prep, key = "datos_train")
datos_test   <- as.h2o(datos_test_prep, key = "datos_test")

# Espacio de búsqueda de cada hiperparámetro
# ==============================================================================
hiperparametros <- list(
  epochs = c(50, 100, 500, 1000),
  hidden = list(5, 10, 25, 50, 100, c(10, 10))
)


# Búsqueda por validación cruzada
# ==============================================================================
variable_respuesta <- 'DCombo'
predictores <- setdiff(colnames(datos_train), variable_respuesta)

grid <- h2o.grid(
  algorithm    = "deeplearning",
  activation   = "Rectifier",
  x            = predictores,
  y            = variable_respuesta,
  training_frame  = datos_train,
  nfolds       = 20, #validacion cruzada
  standardize  = FALSE,
  hyper_params = hiperparametros,
  search_criteria = list(strategy = "Cartesian"),
  seed         = 123,
  grid_id      = "grid"
)

resultados_grid <- h2o.getGrid(
  sort_by = 'rmse',
  grid_id = "grid",
  decreasing = FALSE
)
data.frame(resultados_grid@summary_table)

# Mejor modelo encontrado
# ==============================================================================
modelo_final <- h2o.getModel(resultados_grid@model_ids[[24]])

predicciones <- h2o.predict(
  object  = modelo_final,
  newdata = datos_test
)

predicciones <- predicciones %>%
  as_tibble() %>%
  mutate(valor_real = as.vector(datos_test$DCombo))

predicciones %>% head(5)

#rmse(predicciones, truth = Combo, estimate = predict, na_rm = TRUE)

modelo_final@allparameters
mojo_destination <- h2o.save_mojo(modelo_final, path = getwd(),filename = "modelo_DCombo")

# Se apaga el cluster H2O
h2o.shutdown(prompt = FALSE)

(sum(predicciones[,1]==predicciones[,7]))/dim(predicciones)[1]

