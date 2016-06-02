data {
  int<lower=0> g; // number of groups
  int<lower=0> n; // number of samples
  int<lower=0> k; // number of attributes
  matrix[n, k] x; // samples
  vector[n] y; // targets
  int idx[n]; // group indexes of samples
  vector[k] zeros; // vector of zeros

  int<lower=0> n_new; // number of new samples
  matrix[n_new, k] x_new; // new samples
  int<lower=0> group; // predicting for which group
}

parameters {
  cholesky_factor_corr[k] L_Omega; // prior correlation
  vector<lower=0>[k] tau; // prior scale
  vector[k] betas[g]; // ind. coeffs
  vector[g] alpha; // intercept
  real<lower=0> sigma;
}
model {
  tau ~ cauchy(0, 2.5);
  L_Omega ~ lkj_corr_cholesky(2);
  betas ~ multi_normal_cholesky(zeros, diag_pre_multiply(tau,L_Omega));

  for (i in 1:n){
    y[i] ~ normal(x[i] * betas[idx[i]] + alpha[idx[i]], sigma);
  }
}

generated quantities {
  vector[n_new] y_new;
  for (i in 1:n_new){
    y_new[i] <- normal_rng(x_new[i] * betas[group] + alpha[group], sigma);
  }
}
