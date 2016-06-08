data {
  int<lower=0> n; // number of samples
  int<lower=0> k; // number of attributes
  matrix[n, k] x; // attributes
  vector[n] y; // values
  
  int<lower=0> n_new; // number of predicting samples
  matrix[n_new, k] x_new;
}

parameters {
  real alpha; // intercept
  vector[k] beta;
  real<lower=0, upper=1000> sigma;
}

model {
  y ~ normal(x * beta + alpha, sigma);
}

generated quantities {
  vector[n_new] y_new;
  for (i in 1:n_new)
    y_new[i] <- normal_rng(x_new[i] * beta + alpha, sigma);
}