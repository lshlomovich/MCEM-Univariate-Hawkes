function [time_points, ith_spike_loc1, log_ll] = disc_time_hp_grid(counts_mat, n_repetitions, estimated_params, bin_width, p)
% OTHER OUTPUTS: time_points, S_iter, ith_spike_loc1, log_likelihood, 
% This function will, given a univariate counts matrix, essentially
% simulate a process, conditioning on the location of each successive event
% time. That is, we can generate a timepoint based on the CDF of the next
% inter-arrival time, and then condition (or truncate) such that it occurs
% in the bin it has been observed in. This is done for a given [v, A, B]
% We can then aim to repeat this many times and get the mean likelihood.
% This is then what will be minimised. 

log_ll = 0; 
log_ll_alt = 0; 
values_list = [];
% Set up needed variables from the input
% estimated_params is p*(2*p+1)
v = estimated_params(:,1);          % extract baseline, v 
A = estimated_params(:,2:2+(p-1));  % extract A, excitation (matrix if p>1)
B = estimated_params(:,2*p+1:end);  % extract B, decay term (matrix if p>1)

% Initialise counts with a section from the provided counts matrix
counts = counts_mat(:,1:p);
n_events = sum(counts,1);           % The total number of events per proc.  
%log_likelihood = zeros(n_repetitions,1);

% Get the location of the ith spike in terms of the bin number (UNIVARIATE)
ith_spike_loc1 = zeros(n_events(1),1);
cnt_index = 1;
get_nonzeros1 = find(counts(:,1)); 
% Loop over each non-empty bin
for tp_index = 1:length(get_nonzeros1)
    % Get the index of the non-empty bin being considered
    tp = get_nonzeros1(tp_index);
    % Get the number of counts in that bin
    num_cnts = counts(tp,1);
    % Fill in that the ith to the i+num_cnts th event are in bin tp
    ith_spike_loc1(cnt_index:cnt_index+num_cnts-1)=repmat(tp,num_cnts,1);  % not sure why this used to be tp-1? 
    cnt_index = cnt_index + num_cnts;
end
    for i = 1:n_repetitions
        % UNIVARIATE SO JUST CONSIDERING PROCESS 1
        time_points = [];
        % Initialise first values
        S_iter(1) = 1;
        A_s(1) = 0;
        fvals = [];
        uniq_spike_loc = unique(ith_spike_loc1);
        for ii = 1:length(uniq_spike_loc) % Loop over the bins with events
            bin_num = uniq_spike_loc(ii);         % Get non-empty bin
            binub = bin_width*bin_num;            % Lb of bin
            binlb = bin_width*(bin_num-1);        % Ub of bin
            % Get the number of events in the given bin
            total_in_bin = length(ith_spike_loc1(ith_spike_loc1==bin_num));
            if total_in_bin > 0 %%%%%%%%%% can use 0 or 1 here, 0 preferred
                % STARTING POINTS
                rand_start = bin_width*sort(rand(1,total_in_bin));
                % OPTIONS FOR MINIMISE FN
                options5 = optimoptions(@fmincon,'Display','off','MaxFunctionEvaluations',1e5,'StepTolerance',1e-5,'Algorithm','interior-point'); 
                
                [bin_times,fval] = fmincon(@(theta) max_cond_pdf(theta,time_points,bin_width,binub,v,A,B), rand_start+binlb,[],[],[],[],[],[],[],options5);
                               
                fvals = [fvals;fval];
                % ADDITION OF LIKILHOOD COMPUTATION
                [val1,val2] = max_cond_pdf(bin_times, time_points, bin_width, binub, v,A,B);
                values_list = [values_list;val2];
                log_ll = log_ll + log(val2);
                log_ll_alt = log_ll_alt + log(-val1);
                time_points = [time_points;bin_times']; 
                
                % Update S_iter for the next step
                if length(time_points)==length(bin_times)
                    jj_start = 2;     
                else
                    jj_start = 1;
                end
                for jj=jj_start:total_in_bin
                    S_iter = [S_iter;exp(-B*(time_points(end-total_in_bin+jj)-time_points(end-total_in_bin+jj-1)))*S_iter(end) + 1];
                    % As in Ozaki (1979)
                    A_s = [A_s;sum(exp(-B*(time_points(end-total_in_bin+jj)-time_points(1:end-total_in_bin+jj-1))))];  
                end
            else
                % If we're generating the first timepoint:
                if isempty(time_points)
                    time_points = -(1/v)*log(exp(-v*binlb) - rand(1)*(exp(-v*binlb) - exp(-v*binub)));
                    % ADDITION OF LIKELIHOOD COMPUTATION
                    new_fval = v*exp(-v*time_points);
                    log_ll = log_ll + log(new_fval);
                else
                    % Use inv cdf
                    F_ub_tp = 1-exp(-( v*(binub-time_points(end)) + A/B * S_iter(end) * (1 - exp(-B*(binub-time_points(end)))) ));
                    F_lb_tp = 1-exp(-( v*(binlb-time_points(end)) + A/B * S_iter(end) * (1 - exp(-B*(binlb-time_points(end)))) ));

                    exp_F_ub_tp = exp(-( v*(binub-time_points(end)) + A/B * S_iter(end) * (1 - exp(-B*(binub-time_points(end)))) )); %exp(sym(-( v*(binub-time_points(end)) + A/B * S_iter(end) * (1 - exp(-B*(binub-time_points(end)))) )));
                    exp_F_lb_tp = exp(-( v*(binlb-time_points(end)) + A/B * S_iter(end) * (1 - exp(-B*(binlb-time_points(end)))) )); %exp(sym(-( v*(binlb-time_points(end)) + A/B * S_iter(end) * (1 - exp(-B*(binlb-time_points(end)))) )));
                    c = exp_F_lb_tp - exp_F_ub_tp; %F_ub_tp - F_lb_tp;
                    r = rand(1);
                    K = -log(-r*c + exp_F_lb_tp) - (A/B)*S_iter(end);
                    old_test_time = time_points(end)-log(rand(1))/v;
                    tol = 1;
                    while tol > 0.0001  %vpa(tol) % if sym is used
                        log_U = log(-r*c + exp_F_lb_tp); 
                        func_val = log_U + v*(old_test_time-time_points(end)) + A/B*S_iter(end)*(1-exp(-B*(old_test_time - time_points(end))));
                        diff_func_val = v + A*S_iter(end)*exp(-B*(old_test_time - time_points(end)));
                        test_time = old_test_time - (func_val)/(diff_func_val);
                        tol = abs(test_time - old_test_time);
                        old_test_time = test_time;
                    end
                    test_time = test_time; %double(vpa(test_time));
                    
                    if isnan(test_time)
                        %disp('NaN time generated by inverse sampling')
                        test_time = bin_width*rand(1)+binlb;
                    end
                    
                    % ADDITION OF LIKELIHOOD COMPUTATION
                    % pdf of single timepoint is f_T(t)/c
                    A_tp = exp(-B*(test_time-time_points(end)))*(1+A_s(end));
                    new_fval = (v+A_tp)*exp(-v*(test_time-time_points(end)))*exp(1./B * (A_tp - A_s(end) -1));
                    log_ll = log_ll + log(new_fval); 
                    
                    time_points = [time_points;test_time];
                    % Update S_iter for the next step
                    for jj=1:total_in_bin
                        S_iter = [S_iter;exp(-B*(time_points(end-total_in_bin+jj)-time_points(end-total_in_bin+jj-1)))*S_iter(end) + 1];
                        % As in Ozaki (1979)
                        A_s = [A_s;sum(exp(-B*(time_points(end-total_in_bin+jj)-time_points(1:end-total_in_bin+jj-1))))];  
                    end
                end

            end
            
        end
    end 
    
    
end


