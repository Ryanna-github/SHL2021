data {
    int<lower=1> N;
    int states[N];
    int have_loc[N];
    int accuracy_level[N];
    int speed_level[N];
    real speed_log[N];
    real acc_wd_std_log[N];
    real speed_wd_max_log[N];
}
parameters {
    simplex[8] theta_states;
    simplex[2] theta_have_loc;
    simplex[3] theta_loc_accuracy_level;
    real<lower=0> mu_speed_wd_max_log;
    real<lower=0> sigma_speed_wd_max_log;
    simplex[3] theta_speed_level;
    real<lower=0> mu_speed_log;
    real<lower=0> sigma_speed_log;
    real<lower=0> mu_accc_wd_std_log;
    real<lower=0> sigma_acc_wd_std_log;
}
model {
    states ~ multinomial(theta_states);
}