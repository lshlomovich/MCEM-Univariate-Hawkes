%% MCEM Function
% UNIVARIATE CASE
% MC_EM for hawkes process param estimation with parameters v,alpha,beta 

function [mean_estimate, params, fval1, grads, hessians] = MCEM_univariate(data, N_monte_carlo, n_times, init_choice, method, seed, end_time, bin_width, tol)

    % data:               the count data to be modelled (column vector)
    % N_monte_carlo:      number of monte carlo samples
    % n_times:            a maximum number of times to repeat the EM 
    %(say 100, will depend on rate of convergence, included as a fail-safe)
    % init_choice         the initial parameter estimate to start
    % minimisation algorithm from, can use sort(rand(1,3)) 
    % method              either 'unif' (much quicker) or 'seq' (slower but less biased)
    % seed:               if wishing to set the seed - 0 if not
    % tol                 tolerance threshold, usually converges within 
    % 1e-2 very quickly but can use 1e-3 and should still be within 50 reps
    % naturally, it will vary from run to run
    
    p = size(data,2);  % will be 1 in the univariate case
    params = zeros(p*n_times,3); 
    
    if seed~= 0
        rng(seed)
    end
    
    params(1,:) = init_choice; % assign random first guess
    
    grads = [];
    hessians = [];

    % Initialise counters and tolerance 
    j=1;
    eps=tol+1;
    
    all_weights_unif = [];
    all_weights_seq = [];
    
    while eps>tol && j<n_times %set some tolerance and do while here
        clear T
        disp(j)
        %EXPECTATION STEP 
        weights_raw = zeros(N_monte_carlo,1);
        for i=1:N_monte_carlo
            if mod(i,10)==0
                disp(i)
            end
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            current_params = params(j*p-p+1:j*p,:);

            baseline_est = current_params(:,1);
            excitation_est = current_params(:,2:2+p-1);
            decay_est = current_params(:,2+p:end);
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Uniform method %
            %[T{i}, log_density] = generate_uniform_times(data,p,bin_width);  
            %weights_raw(i) = -complete_likelihood(T{i},baseline_est,excitation_est,decay_est,end_time,p)-(log_density);
            if strcmp(method,'unif')
                [T(i,:), log_density(i)] = generate_uniform_times(data,p,bin_width);  
                weights_raw(i) = -complete_likelihood_univariate(T(i,:),baseline_est,excitation_est,decay_est,end_time)-(log_density(i));
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Sequential method %
            elseif strcmp(method,'seq')
                [T(i,:), ~, log_density(i)] = disc_time_hp_grid(data, 1, [params(j,1),params(j,2),params(j,3)], bin_width, 1); 
                %Compute the Importance Sampling weights 
                weights_raw(i) = exp((complete_likelihood_univariate(T(i,:),params(j,1),params(j,2),params(j,3),end_time)-log_density(i))); 
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            else
                disp('Choose a valid method')
                return
            end
            
            
            
           
        end
        %% Normalise weights: mean(weights) is the IS estimator for the messy integral (normalisation constant)

        if strcmp(method,'unif')
            % For this method weights need to be rescaled to not be Inf
            weights_raw = exp(weights_raw-min(weights_raw));
        end
        weights_raw_rescaled = weights_raw./sum(weights_raw);

        
        
        %MAXIMISATION STEP
        zhat = params(j,:); 
        options1 = optimoptions('fmincon','MaxIter',2e4,'MaxFunEvals',2e4,'Algo','interior-point','TolFun',1e-10,'TolX',1e-10,'GradObj','off','Hessian','off','Display','off','DerivativeCheck','off');
        myfun = @(z) complete_sum_likelihood_univariate(T, z(:,1),z(:,2:2+p-1),z(:,2+p:end), end_time, weights_raw_rescaled,p); 

        weights1_time=tic; [params(j+1,:), fval1(j),~,~,~,grad,hessian] = fmincon(myfun,zhat,[],[],[],[],[],[],[],options1); w1_time = toc(weights1_time); disp(w1_time)
        
        grads = [grads;grad];
        hessians = [hessians;hessian];
        disp(params(j+1,:))
        eps = norm(params(j+1,:)-params(j,:));
        disp(eps)
        j = j+1; 
    end

    first_zero_row = find(all(params==0,2), 1);
    if isempty(first_zero_row)
        mean_estimate = mean(params(2:end,:));
    else
        mean_estimate = mean(params(2:first_zero_row-1,:));
    end
end

