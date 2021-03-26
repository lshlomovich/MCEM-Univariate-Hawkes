% Example script showing usage of the MCEM algorithm. 
% Repeat for n_realisations of Hawkes processes, say

n_realisations = 10; % Set to whatever you like for you simulation study

for explore_rep = 1:n_realisations
    
    % Set initial parameters
    
    end_time = 1.5e3;  % simulation time
    v = 0.3;  % baseline parameter
    beta=1.1;  % decay parameter
    alpha=0.8;  % excitation parameter
    
    % p = 1 in the univariate case
    p = length(v);
    
    % Number of Monte Carlo repetitions 
    N_monte_carlo = 20; % can work for even as low as this
    % Declare the level of aggregation of your data
    bin_width = 1; 
    % Simulate the Hawkes process 
    [t, ~, ~, ~] = SimulateMarkedHawkes1D(end_time, v, 0, beta, 'const', alpha);
    E = t{:};
    % Aggregate the data
    data = transpose(histcounts(E, 0:bin_width:end_time));
    % Set a maximum number of repeats for the MCEM algorithm, setting this
    % large will ensure convergence rather than termination of the
    % algorithm
    n_times = 50;
    % Get MLE of the simulated times
    options = optimoptions('fmincon','MaxIter',2e4,'MaxFunEvals',2e4,'Algo','interior-point','TolFun',1e-10,'TolX',1e-10,'GradObj','off','Hessian','off','Display','off','DerivativeCheck','off');
    myfunMv_true = @(z) complete_likelihood(E', z(1),z(2),z(3), end_time, p);
    tic; [true_value_est,fval,exitflag,output,lambda,grad,hessian] = fmincon(myfunMv_true, [v,alpha,beta],[],[],[],[],[],[],[],options); toc
    disp('MLE of raw simulated times')
    disp(true_value_est)
    
    % MCEM Algorithm
    % Take the init_choice to be either a guess using another method, or sort(rand(1,3))
    % For example, can run using the uniform method to get an init choice
    % that's sensible, and then run using sequential method, also seq works
    % very well with a random choice also
    
    init_choice = sort(rand(1,3));
    method = 'unif'; 
    [init_est, all_init_ests, ~, ~, ~] = MCEM_univariate(data, N_monte_carlo, n_times, init_choice, method, 0, end_time, bin_width, 1e-2);
    method = 'seq'; 
    [mean_est, ests, ~, ~, ~] = MCEM_univariate(data, N_monte_carlo, n_times, init_est, method, 0, end_time, bin_width, 1e-2);

    true_ests(explore_rep,:) = true_value_est;
    mean_ests(explore_rep,:) = mean_est;

end