data {
  int<lower=0> n; // number of samples
  int<lower=0> k; // number of attributes
  matrix[n, k] x; // attributes
  vector[n] y; // values
  int<lower=0> n_new; // number of predicting samples
  matrix[n_new, k] x_new;
}

parameters {
  real alpha;
  vector[k] beta;
  real<lower=0.0001> shape;
}

model {
    for(i in 1:n)
      y[i] ~ gamma(shape, shape / exp(x[i,] * beta + alpha));
}

generated quantities {
  vector[n_new] y_new;
    for(i in 1:n_new)
      y_new[i] <- gamma_rng(shape, shape / exp(x_new[i,] * beta + alpha));
}

